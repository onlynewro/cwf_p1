"""Command-line entry point for the cosmology joint fit analysis."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from scipy.optimize import differential_evolution

from joint_fit_multiproc import (
    CovarianceComputationError,
    _finite_float_or_none,
    calculate_statistics,
    estimate_covariance,
    total_chi2,
)
from src.data_loaders.bao_loader import BAOData
from src.data_loaders.cmb_loader import CMBData
from src.data_loaders.sne_loader import SNData
from src.models.lcdm import LCDMModel
from src.models.rd_fit_wrapper import RDFitWrapper
from src.models.seven_d import SevenDModel
from src.utils.logging_config import (
    StructuredLogger,
    build_run_metadata,
    compute_sha256,
)
from src.utils.validation import ConfigValidationError, require_existing_file


def _safe_load_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigValidationError(
            f"YAML file {path} must contain a mapping at the top level"
        )
    return loaded


def _load_configuration(config_path):
    resolved = require_existing_file(
        config_path,
        description='analysis configuration file'
    )
    config_dir = Path(resolved).parent
    config_data = _safe_load_yaml(resolved)
    config_data['config_path'] = str(resolved)
    config_data['config_dir'] = str(config_dir)

    paths_file = config_data.get('paths_file')
    paths_data = {}
    if paths_file:
        resolved_paths = require_existing_file(
            paths_file,
            base_dir=config_dir,
            description='paths configuration file'
        )
        paths_dir = Path(resolved_paths).parent
        raw_paths = _safe_load_yaml(resolved_paths)
        for name, section in raw_paths.items():
            if not isinstance(section, dict):
                raise ConfigValidationError(
                    f"Section '{name}' in {resolved_paths} must be a mapping"
                )
            section_copy = copy.deepcopy(section)
            base_dir_value = section_copy.get('base_dir')
            if base_dir_value is None:
                resolved_base = paths_dir
            else:
                resolved_base = (paths_dir / base_dir_value).resolve()
            section_copy['base_dir'] = str(resolved_base)
            paths_data[name] = section_copy

    config_data['paths'] = paths_data
    return config_data


def _build_parsers():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to the YAML configuration file'
    )

    parser = argparse.ArgumentParser(
        description='7D Cosmology Joint Fitting',
        parents=[base_parser]
    )
    parser.add_argument('--model', choices=['LCDM', '7D', 'both'],
                        help='Model to fit (default from configuration)')
    parser.add_argument('--bao-file', type=str,
                        help='BAO data file (JSON)')
    parser.add_argument('--sn-file', type=str,
                        help='Supernova data file')
    parser.add_argument('--use-cmb', action='store_true',
                        help='Include CMB constraints')
    parser.add_argument('--use-default-bao', action='store_true',
                        help='Load the packaged DESI BAO catalogue when no file is supplied')
    parser.add_argument('--rd-mode', choices=['fixed', 'fit'],
                        help='Sound horizon mode (default from configuration)')
    parser.add_argument('--disable-bao-cov', action='store_true',
                        help='Ignore supplied BAO covariance matrices')
    parser.add_argument('--no-proxy', action='store_true',
                        help='Omit the legacy QSO proxy point from the default BAO catalogue')
    parser.add_argument('--drop-lya-dh', action='store_true',
                        help='Remove the Lyα DH/rd point from the final fit')
    parser.add_argument('--diagnose-lya-dh', action='store_true',
                        help='Temporarily inspect residuals without the Lyα DH/rd point before fitting')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of workers for parallelization (single-worker mode enforced)'
    )
    parser.add_argument('--maxiter', type=int,
                        help='Maximum iterations for optimization')
    parser.add_argument('--output', type=str,
                        help='Output file for results')
    return base_parser, parser


def _merge_section(config_data, section_name):
    merged = {}
    paths_section = config_data.get('paths', {}).get(section_name)
    if isinstance(paths_section, dict):
        merged = copy.deepcopy(paths_section)
    section_override = config_data.get(section_name)
    if isinstance(section_override, dict):
        for key, value in section_override.items():
            merged[key] = value
    if 'base_dir' not in merged and 'config_dir' in config_data:
        merged['base_dir'] = config_data['config_dir']
    return merged


def _log_dataset_summary(
    logger: StructuredLogger,
    name: str,
    message: str,
    payload: Dict[str, object],
    level: int = logging.INFO,
) -> None:
    logger.log_event(f'data_load.{name}', payload, level=level, message=message)


def _summarize_bao(bao_data: BAOData, source: str, use_covariance: bool, include_proxy: bool) -> Dict[str, object]:
    return {
        'loaded': True,
        'source': source,
        'observables': bao_data.count_observables(),
        'entry_count': len(bao_data.data),
        'covariance_entries': bao_data.covariance_entry_count(),
        'use_official_covariance': use_covariance,
        'include_proxy': include_proxy,
    }


def _ensure_workers(args):
    requested = args.workers if args.workers is not None else 1
    if requested <= 0:
        requested = 1
    args.requested_workers = requested
    args.workers = 1
    return args


def main():
    base_parser, parser = _build_parsers()
    preliminary_args, remaining = base_parser.parse_known_args()

    try:
        config_data = _load_configuration(preliminary_args.config)
    except ConfigValidationError as exc:
        print(f"Configuration error: {exc}")
        return {}

    run_defaults = config_data.get('run', {})
    bao_defaults = _merge_section(config_data, 'bao')
    cmb_defaults = _merge_section(config_data, 'cmb')

    parser.set_defaults(
        model=run_defaults.get('model', 'both'),
        rd_mode=run_defaults.get('rd_mode', 'fixed'),
        workers=run_defaults.get('workers', 1),
        maxiter=run_defaults.get('maxiter', 1000),
        output=run_defaults.get('output', 'fit_results.json'),
        use_cmb=cmb_defaults.get('enabled', False),
        disable_bao_cov=not bao_defaults.get('use_official_covariance', True),
        no_proxy=not bao_defaults.get('include_proxy', True),
    )
    parser.set_defaults(config=preliminary_args.config)

    args = parser.parse_args(remaining)
    args = _ensure_workers(args)

    run_logger = StructuredLogger(run_id=run_defaults.get('run_id'))
    run_logger.log_event(
        'run_start',
        {
            'config_path': args.config,
            'model': args.model,
            'rd_mode': args.rd_mode,
            'requested_workers': getattr(args, 'requested_workers', args.workers),
            'workers': args.workers,
            'parallel_mode': 'single_worker',
        },
        message='=== 7D Cosmology Joint Fitting (single-worker execution) ==='
    )
    run_logger.log_event(
        'runtime_configuration',
        {
            'save_intermediate': bool(run_defaults.get('save_intermediate', False)),
            'output': args.output,
            'requested_workers': getattr(args, 'requested_workers', args.workers),
            'parallel_mode': 'single_worker',
        },
        message='Running in single-worker mode; parallel computation is disabled'
    )

    bao_config = _merge_section(config_data, 'bao')
    sn_config = _merge_section(config_data, 'sn')
    cmb_config = _merge_section(config_data, 'cmb')

    use_covariance = bao_config.get('use_official_covariance', True)
    if args.disable_bao_cov:
        use_covariance = False
    bao_config['use_official_covariance'] = use_covariance

    include_proxy = bao_config.get('include_proxy', True)
    if args.no_proxy:
        include_proxy = False
    bao_config['include_proxy'] = include_proxy

    save_intermediate = bool(run_defaults.get('save_intermediate', False))

    run_logger.log_event(
        'data_load.start',
        {
            'bao_requested': bool(args.bao_file or args.use_default_bao or bao_config.get('data_file') or bao_config.get('datasets')),
            'sn_requested': bool(args.sn_file or sn_config.get('file')),
            'use_cmb': bool(args.use_cmb),
        },
        message='Loading datasets...'
    )

    datasets = {'bao': None, 'sn': None, 'cmb': None}
    dataset_summary: Dict[str, object] = {}
    checksum_entries: Dict[str, str] = {}

    if config_data.get('config_path'):
        config_checksum = compute_sha256(Path(config_data['config_path']))
        if config_checksum:
            checksum_entries['config'] = config_checksum

    bao_requested = bool(
        args.bao_file
        or args.use_default_bao
        or bao_config.get('data_file')
        or bao_config.get('datasets')
    )
    bao_source = None
    if bao_requested:
        bao_source = args.bao_file or bao_config.get('data_file') or 'configuration datasets'
        try:
            bao_data = BAOData(
                args.bao_file if args.bao_file else None,
                use_official_covariance=use_covariance,
                include_proxy=include_proxy,
                config=bao_config,
            )
        except ConfigValidationError as exc:
            _log_dataset_summary(
                run_logger,
                'bao.error',
                f"BAO: configuration error ({exc})",
                {'error': str(exc)},
                level=logging.ERROR,
            )
            dataset_summary['bao'] = {'loaded': False, 'error': str(exc)}
            datasets['bao'] = None
        except Exception as exc:  # pylint: disable=broad-except
            _log_dataset_summary(
                run_logger,
                'bao.load_failed',
                f"BAO: failed to load ({exc})",
                {'error': str(exc)},
                level=logging.ERROR,
            )
            dataset_summary['bao'] = {'loaded': False, 'error': str(exc)}
            datasets['bao'] = None
        else:
            if args.disable_bao_cov:
                bao_data.remove_all_covariances()
            datasets['bao'] = bao_data
            payload = _summarize_bao(bao_data, bao_source, use_covariance, include_proxy)
            _log_dataset_summary(
                run_logger,
                'bao.success',
                f"BAO: {payload['observables']} observables from {payload['entry_count']} entries loaded",
                payload,
            )
            dataset_summary['bao'] = payload
            for idx, file_path in enumerate(bao_data.source_files):
                digest = compute_sha256(Path(file_path))
                if digest:
                    checksum_entries[f'bao_source_{idx}'] = digest
    else:
        _log_dataset_summary(
            run_logger,
            'bao.skipped',
            'BAO: skipped (no configuration or file provided)',
            {},
        )
        dataset_summary['bao'] = {'loaded': False, 'reason': 'not requested'}

    sn_file = args.sn_file if args.sn_file else sn_config.get('file')
    sn_marginalize = sn_config.get('marginalize_m', True)
    if sn_file:
        try:
            sn_data = SNData(sn_file, marginalize_m=sn_marginalize, config=sn_config)
            datasets['sn'] = sn_data
            sn_summary = sn_data.summary()
            sn_summary['loaded'] = True
            dataset_summary['sn'] = sn_summary
            message = (
                f"SN: {sn_summary['count']} points loaded (cov rank = {sn_summary['cov_rank']})"
            )
            details = {
                'file': sn_summary['file'],
                'columns': sn_summary['columns'],
                'covariance_source': sn_summary['cov_source'],
            }
            _log_dataset_summary(run_logger, 'sn.success', message, details)
            if sn_summary['file']:
                digest = compute_sha256(Path(sn_summary['file']))
                if digest:
                    checksum_entries['sn'] = digest
            if sn_summary['cov_source']:
                digest = compute_sha256(Path(sn_summary['cov_source']))
                if digest:
                    checksum_entries['sn_covariance'] = digest
        except ConfigValidationError as exc:
            _log_dataset_summary(
                run_logger,
                'sn.config_error',
                f"SN configuration error ({exc})",
                {'error': str(exc)},
                level=logging.WARNING,
            )
            dataset_summary['sn'] = {'loaded': False, 'error': str(exc)}
            datasets['sn'] = None
        except Exception as exc:  # pylint: disable=broad-except
            _log_dataset_summary(
                run_logger,
                'sn.load_failed',
                f"SN data load failed ({exc})",
                {'error': str(exc)},
                level=logging.WARNING,
            )
            dataset_summary['sn'] = {'loaded': False, 'error': str(exc)}
            datasets['sn'] = None
    else:
        dataset_summary['sn'] = {'loaded': False, 'reason': 'not requested'}

    if args.use_cmb:
        datasets['cmb'] = CMBData(config=cmb_config)
        dataset_summary['cmb'] = {
            'loaded': True,
            'observables': datasets['cmb'].count_observables(),
        }
        _log_dataset_summary(
            run_logger,
            'cmb.success',
            'CMB: constraints loaded',
            dataset_summary['cmb'],
        )
    else:
        datasets['cmb'] = None
        dataset_summary['cmb'] = {'loaded': False, 'observables': 0, 'enabled': False}
        _log_dataset_summary(
            run_logger,
            'cmb.skipped',
            'CMB: constraints not included',
            dataset_summary['cmb'],
        )

    if datasets['bao'] is not None and datasets['bao'].count_observables() > 0:
        fid_lcdm = LCDMModel()
        fid_params = [0.67, 0.31]
        run_logger.log_event(
            'bao_fiducial_check.start',
            {'model': 'LCDM', 'params': fid_params},
            message='Running fiducial BAO sanity check (LCDM: h=0.67, Ωm=0.31).'
        )
        fid_df, fid_max_pull = datasets['bao'].print_residual_table(
            fid_lcdm,
            fid_params,
            rd_value=147.0,
            return_dataframe=True,
        )
        if save_intermediate and fid_df is not None:
            path = run_logger.save_dataframe('diagnostics/fiducial_bao_residuals.csv', fid_df)
            if path:
                run_logger.log_event(
                    'artifact_saved',
                    {'path': str(path)},
                    message=f'Saved fiducial BAO residuals to {path.name}'
                )
        run_logger.log_event(
            'bao_fiducial_check.complete',
            {'max_abs_pull': fid_max_pull},
            message=f'Fiducial maximum |pull| = {fid_max_pull:.3f}'
        )

        max_frac = 0.0
        for point in datasets['bao'].data:
            obs_vec, theory_vec, _ = datasets['bao']._collect_observables(point, fid_lcdm, fid_params, 147.0)
            if obs_vec is None or obs_vec.size == 0:
                continue
            frac_diff = np.abs(obs_vec - theory_vec) / obs_vec
            if frac_diff.size:
                max_frac = max(max_frac, float(np.max(frac_diff)))
            if np.any(frac_diff > 0.01):
                run_logger.log_event(
                    'bao_fiducial_check.warning',
                    {
                        'entry': point.get('name', 'BAO_point'),
                        'fractional_difference': frac_diff.tolist(),
                    },
                    level=logging.WARNING,
                    message=f"BAO entry {point.get('name', 'BAO_point')} exceeds 1% fractional difference"
                )
        if max_frac <= 0.01:
            run_logger.log_event(
                'bao_fiducial_check.summary',
                {'max_fractional_difference': max_frac},
                message='All fiducial predictions agree within 1% of observations'
            )

        if args.diagnose_lya_dh:
            diagnostic_flag = False
            with datasets['bao'].temporarily_drop('DESI Lyα GCcomb', 'DH_over_rd') as dropped:
                diagnostic_flag = dropped
                if dropped:
                    diag_df, diag_pull = datasets['bao'].print_residual_table(
                        fid_lcdm,
                        fid_params,
                        rd_value=147.0,
                        title='(Lyα DH/rd temporarily removed)',
                        return_dataframe=True,
                    )
                    if save_intermediate and diag_df is not None:
                        path = run_logger.save_dataframe('diagnostics/fiducial_without_lya_dh.csv', diag_df)
                        if path:
                            run_logger.log_event(
                                'artifact_saved',
                                {'path': str(path)},
                                message=f'Saved diagnostic BAO residuals to {path.name}'
                            )
                    run_logger.log_event(
                        'bao_diagnostic.lya_dh_removed',
                        {'max_abs_pull': diag_pull},
                        message=f'Diagnostic maximum |pull| (Lyα DH removed) = {diag_pull:.3f}'
                    )
                else:
                    run_logger.log_event(
                        'bao_diagnostic.lya_dh_missing',
                        {},
                        level=logging.WARNING,
                        message='Diagnostic request: Lyα DH/rd observable unavailable for removal.'
                    )
            if diagnostic_flag:
                run_logger.log_event(
                    'bao_diagnostic.lya_dh_restored',
                    {},
                    message='Lyα DH/rd observable restored for covariance-weighted fitting.'
                )

    if args.drop_lya_dh and datasets['bao'] is not None:
        dropped_final = datasets['bao'].drop_observable('DESI Lyα GCcomb', 'DH_over_rd')
        if dropped_final:
            run_logger.log_event(
                'bao.modify.drop_lya_dh',
                {},
                message='BAO: Lyα DH/rd observable removed from the final fit.'
            )
        else:
            run_logger.log_event(
                'bao.modify.drop_lya_dh_missing',
                {},
                level=logging.WARNING,
                message='BAO: Lyα DH/rd observable not found or already removed.'
            )
    elif args.drop_lya_dh and datasets['bao'] is None:
        run_logger.log_event(
            'bao.modify.drop_lya_dh_unavailable',
            {},
            level=logging.WARNING,
            message='Cannot drop Lyα DH/rd because BAO data are not loaded.'
        )

    final_bao_count = datasets['bao'].count_observables() if datasets['bao'] is not None else 0
    final_bao_entries = len(datasets['bao'].data) if datasets['bao'] else 0
    final_cov_entries = datasets['bao'].covariance_entry_count() if datasets['bao'] is not None else 0
    run_logger.log_event(
        'bao.final_summary',
        {
            'observables': final_bao_count,
            'entry_count': final_bao_entries,
            'covariance_entries': final_cov_entries,
        },
        message=f"BAO (final): {final_bao_count} observables from {final_bao_entries} entries in use"
    )

    n_data = 0
    if datasets['bao'] is not None:
        n_data += datasets['bao'].count_observables()
    if datasets['sn']:
        n_data += datasets['sn'].count_points()
    if datasets['cmb']:
        n_data += datasets['cmb'].count_observables()

    models_to_fit = []
    if args.model in ['LCDM', 'both']:
        models_to_fit.append(LCDMModel())
    if args.model in ['7D', 'both']:
        models_to_fit.append(SevenDModel())

    if args.rd_mode == 'fit':
        models_to_fit = [RDFitWrapper(model) for model in models_to_fit]

    results = {}

    for model in models_to_fit:
        run_logger.log_event(
            'model_fit.start',
            {
                'model': model.name,
                'parameters': list(model.param_names),
                'bounds': list(model.bounds),
            },
            message=f"Fitting {model.name} model..."
        )

        obj = partial(total_chi2,
                     datasets=datasets,
                     model=model,
                     rd_mode=args.rd_mode)

        try:
            de_res = differential_evolution(
                obj,
                bounds=model.bounds,
                workers=1,
                updating='deferred',
                polish=True,
                disp=False,
                maxiter=args.maxiter,
                seed=42,
                tol=1e-6,
                atol=1e-8
            )

            theta_hat = np.asarray(de_res.x, dtype=float)
            param_names = list(model.param_names)
            dof_candidate = int(n_data - len(param_names))

            covariance_payload = None
            correlation_payload = None
            errors_payload = None

            if not de_res.success:
                run_logger.log_event(
                    'model_fit.warning',
                    {'model': model.name, 'message': str(de_res.message)},
                    level=logging.WARNING,
                    message='Optimization did not converge; skipping covariance estimation.'
                )
            elif dof_candidate <= 0:
                run_logger.log_event(
                    'model_fit.warning',
                    {'model': model.name, 'degrees_of_freedom': dof_candidate},
                    level=logging.WARNING,
                    message='Skipping covariance estimation because degrees of freedom are non-positive.'
                )
            else:
                try:
                    cov_result = estimate_covariance(
                        func=obj,
                        theta=theta_hat,
                        param_names=param_names,
                        bounds=getattr(model, 'bounds', None),
                        dof=dof_candidate,
                    )
                    covariance_payload = {
                        'param_names': param_names,
                        'matrix': cov_result.matrix.tolist(),
                        'condition_number': float(cov_result.condition_number),
                    }
                    correlation_payload = {
                        'param_names': param_names,
                        'matrix': cov_result.correlation.tolist(),
                    }
                    errors_payload = {
                        key: _finite_float_or_none(value)
                        for key, value in cov_result.errors.items()
                    }
                except CovarianceComputationError as exc:
                    run_logger.log_event(
                        'model_fit.covariance_failed',
                        {'model': model.name, 'error': str(exc)},
                        level=logging.WARNING,
                        message='Covariance estimation failed.'
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    run_logger.log_event(
                        'model_fit.covariance_exception',
                        {'model': model.name, 'error': str(exc)},
                        level=logging.WARNING,
                        message='Unexpected error during covariance estimation.'
                    )

            stats = calculate_statistics(de_res.fun, n_data, len(model.param_names))

            best_fit_parameters = {
                name: float(value)
                for name, value in zip(model.param_names, de_res.x)
            }

            stats_native = {
                'chi2': _finite_float_or_none(stats.get('chi2')),
                'dof': int(stats.get('dof', 0)),
                'chi2_red': _finite_float_or_none(stats.get('chi2_red')),
                'aic': _finite_float_or_none(stats.get('aic')),
                'bic': _finite_float_or_none(stats.get('bic')),
                'n_data': int(stats.get('n_data', n_data)),
                'n_params': int(stats.get('n_params', len(model.param_names)))
            }

            results[model.name] = {
                'success': bool(de_res.success),
                'bestfit': best_fit_parameters,
                'parameters': best_fit_parameters.copy(),
                'chi2': stats_native['chi2'],
                'dof': stats_native['dof'],
                'chi2_red': stats_native['chi2_red'],
                'aic': stats_native['aic'],
                'bic': stats_native['bic'],
                'n_data': stats_native['n_data'],
                'n_params': stats_native['n_params'],
                'statistics': stats_native,
                'message': str(de_res.message),
                'nfev': int(de_res.nfev),
                'errors': errors_payload,
                'covariance': covariance_payload,
                'correlation': correlation_payload
            }

            run_logger.log_event(
                'model_fit.complete',
                {
                    'model': model.name,
                    'success': bool(de_res.success),
                    'message': str(de_res.message),
                    'nfev': int(de_res.nfev),
                    'statistics': stats_native,
                },
                message=f"Optimization {'succeeded' if de_res.success else 'failed'} for {model.name}"
            )
            bestfit_msg = ', '.join(f"{name}={value:.6f}" for name, value in zip(model.param_names, de_res.x))
            run_logger.log_event(
                'model_fit.best_parameters',
                {'model': model.name, 'parameters': best_fit_parameters},
                message=f"{model.name} best-fit parameters: {bestfit_msg}"
            )

            for (name, value), (low, high) in zip(zip(model.param_names, de_res.x), model.bounds):
                span = high - low
                if span <= 0:
                    continue
                lower_frac = (value - low) / span
                upper_frac = (high - value) / span
                if lower_frac < 0.05 or upper_frac < 0.05:
                    run_logger.log_event(
                        'model_fit.parameter_near_bounds',
                        {
                            'model': model.name,
                            'parameter': name,
                            'value': float(value),
                            'bounds': [float(low), float(high)],
                        },
                        level=logging.WARNING,
                        message=f"Parameter '{name}' is within 5% of its bounds ({low}, {high})."
                    )

            run_logger.log_event(
                'model_fit.statistics',
                {'model': model.name, **stats_native},
                message=(
                    f"{model.name} statistics: χ²={stats_native['chi2']}, χ²/dof={stats_native['chi2_red']}, "
                    f"AIC={stats_native['aic']}, BIC={stats_native['bic']}"
                )
            )

            if save_intermediate:
                if covariance_payload:
                    cov_path = run_logger.save_json(
                        Path('covariance') / f"{model.name.lower()}_covariance.json",
                        covariance_payload,
                    )
                    run_logger.log_event(
                        'artifact_saved',
                        {'path': str(cov_path)},
                        message=f'Saved covariance estimate to {cov_path.name}'
                    )
                if correlation_payload:
                    corr_path = run_logger.save_json(
                        Path('covariance') / f"{model.name.lower()}_correlation.json",
                        correlation_payload,
                    )
                    run_logger.log_event(
                        'artifact_saved',
                        {'path': str(corr_path)},
                        message=f'Saved correlation matrix to {corr_path.name}'
                    )
                if errors_payload:
                    err_path = run_logger.save_json(
                        Path('covariance') / f"{model.name.lower()}_errors.json",
                        errors_payload,
                    )
                    run_logger.log_event(
                        'artifact_saved',
                        {'path': str(err_path)},
                        message=f'Saved parameter errors to {err_path.name}'
                    )

            if datasets['bao'] is not None and datasets['bao'].count_observables() > 0:
                rd_for_residuals = 147.0
                if args.rd_mode == 'fit' and 'rd' in model.param_names:
                    rd_index = model.param_names.index('rd')
                    rd_candidate = de_res.x[rd_index]
                    if np.isfinite(rd_candidate) and rd_candidate > 0:
                        rd_for_residuals = rd_candidate
                residual_df, best_pull = datasets['bao'].print_residual_table(
                    model,
                    de_res.x,
                    rd_value=rd_for_residuals,
                    return_dataframe=True,
                )
                if save_intermediate and residual_df is not None:
                    path = run_logger.save_dataframe(
                        Path('diagnostics') / f"{model.name.lower()}_bao_residuals.csv",
                        residual_df,
                    )
                    if path:
                        run_logger.log_event(
                            'artifact_saved',
                            {'path': str(path)},
                            message=f'Saved {model.name} BAO residuals to {path.name}'
                        )
                run_logger.log_event(
                    'bao_residuals.summary',
                    {'model': model.name, 'max_abs_pull': best_pull},
                    message=f"Maximum |pull| at best-fit {model.name}: {best_pull:.3f}"
                )

        except Exception as exc:  # pylint: disable=broad-except
            run_logger.log_event(
                'model_fit.error',
                {'model': model.name, 'error': str(exc)},
                level=logging.ERROR,
                message=f'ERROR during optimization for {model.name}: {exc}'
            )
            results[model.name] = {
                'success': False,
                'error': str(exc)
            }

    if 'LCDM' in results and '7D' in results:
        if results['LCDM'].get('success') and results['7D'].get('success'):
            delta_aic = results['7D']['statistics']['aic'] - results['LCDM']['statistics']['aic']
            delta_bic = results['7D']['statistics']['bic'] - results['LCDM']['statistics']['bic']
            run_logger.log_event(
                'model_comparison',
                {
                    'delta_aic': delta_aic,
                    'delta_bic': delta_bic,
                },
                message=f"Model comparison ΔAIC (7D-ΛCDM) = {delta_aic:.3f}, ΔBIC = {delta_bic:.3f}"
            )

    output_relative = Path(args.output)
    results_path = run_logger.save_json(output_relative, results)
    run_logger.log_event(
        'results_saved',
        {'path': str(results_path)},
        message=f'Results saved to {results_path}'
    )

    legacy_output_path = Path(args.output)
    try:
        legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
        with legacy_output_path.open('w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
    except Exception as exc:  # pylint: disable=broad-except
        run_logger.log_event(
            'results_legacy_save_failed',
            {'path': str(legacy_output_path), 'error': str(exc)},
            level=logging.WARNING,
            message=f'Warning: Could not save results copy to {legacy_output_path}: {exc}'
        )
    else:
        run_logger.log_event(
            'results_legacy_saved',
            {'path': str(legacy_output_path.resolve())},
            message=f'Legacy results copy saved to {legacy_output_path}'
        )

    metadata = build_run_metadata(
        run_logger,
        arguments=vars(args),
        config_snapshot=config_data,
        dataset_summary=dataset_summary,
        results_path=results_path,
        checksums=checksum_entries,
        extra={'models': list(results.keys())},
    )
    metadata_path = run_logger.save_json('run_metadata.json', metadata)
    run_logger.log_event(
        'metadata_saved',
        {'path': str(metadata_path)},
        message=f'Run metadata saved to {metadata_path}'
    )

    run_logger.log_event(
        'run_complete',
        {'models': list(results.keys())},
        message='Run completed.'
    )

    return results


def cli():
    if sys.platform.startswith('win'):
        mp.freeze_support()
    try:
        main()
    except ConfigValidationError as exc:
        print(f"\nConfiguration error: {exc}")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nFATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()
