"""Command-line entry point for the cosmology joint fit analysis."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from functools import partial

import numpy as np
from scipy.optimize import differential_evolution

from src.data_loaders.bao_loader import BAOData
from src.data_loaders.cmb_loader import CMBData
from src.data_loaders.sne_loader import SNData
from src.models.lcdm import LCDMModel
from src.models.rd_fit_wrapper import RDFitWrapper
from src.models.seven_d import SevenDModel
from joint_fit_multiproc import (
    CovarianceComputationError,
    _finite_float_or_none,
    calculate_statistics,
    estimate_covariance,
    total_chi2,
)


def main():
    parser = argparse.ArgumentParser(description='7D Cosmology Joint Fitting')
    parser.add_argument('--model', choices=['LCDM', '7D', 'both'], default='both',
                        help='Model to fit')
    parser.add_argument('--bao-file', type=str, default=None,
                        help='BAO data file (JSON)')
    parser.add_argument('--sn-file', type=str, default=None,
                        help='Supernova data file')
    parser.add_argument('--use-cmb', action='store_true',
                        help='Include CMB constraints')
    parser.add_argument('--use-default-bao', action='store_true',
                        help='Load the packaged DESI BAO catalogue when no file is supplied')
    parser.add_argument('--rd-mode', choices=['fixed', 'fit'], default='fixed',
                        help='Sound horizon mode')
    parser.add_argument('--disable-bao-cov', action='store_true',
                        help='Ignore supplied BAO covariance matrices')
    parser.add_argument('--no-proxy', action='store_true',
                        help='Omit the legacy QSO proxy point from the default BAO catalogue')
    parser.add_argument('--drop-lya-dh', action='store_true',
                        help='Remove the Lyα DH/rd point from the final fit')
    parser.add_argument('--diagnose-lya-dh', action='store_true',
                        help='Temporarily inspect residuals without the Lyα DH/rd point before fitting')
    parser.add_argument('--workers', type=int, default=-1,
                        help='Number of workers for parallelization (-1 for all CPUs)')
    parser.add_argument('--maxiter', type=int, default=1000,
                        help='Maximum iterations for optimization')
    parser.add_argument('--output', type=str, default='fit_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    if args.workers == -1:
        args.workers = mp.cpu_count()

    print(f"=== 7D Cosmology Joint Fitting ===")
    print(f"Using {args.workers} workers for parallel computation")

    print("\nLoading datasets...")
    datasets = {'bao': None, 'sn': None, 'cmb': None}

    use_covariance = not args.disable_bao_cov
    bao_requested = bool(args.bao_file) or args.use_default_bao
    if bao_requested:
        bao_source = args.bao_file if args.bao_file else 'default DESI catalogue'
        bao_data = BAOData(
            args.bao_file if args.bao_file else None,
            use_official_covariance=use_covariance,
            include_proxy=not args.no_proxy
        )
        if args.disable_bao_cov:
            bao_data.remove_all_covariances()
        datasets['bao'] = bao_data
        print(f"  BAO: {bao_data.count_observables()} observables from {len(bao_data.data)} entries loaded")
        print(f"      source: {bao_source}")
        cov_entries = bao_data.covariance_entry_count()
        if cov_entries:
            print(f"      covariance provided for {cov_entries} BAO entries")
    else:
        print("  BAO: skipped (no file provided)")

    if args.sn_file:
        try:
            sn_data = SNData(args.sn_file)
            datasets['sn'] = sn_data
            sn_summary = sn_data.summary()
            print(f"  SN: {sn_summary['count']} points loaded (cov rank = {sn_summary['cov_rank']})")
            print(f"      file: {sn_summary['file']}")
            columns = sn_summary['columns']
            print(f"      columns: z='{columns.get('z')}', mu='{columns.get('mu')}', sigma='{columns.get('sigma')}'")
            cov_source = sn_summary['cov_source']
            if cov_source:
                print(f"      covariance: {cov_source}")
            else:
                print("      covariance: none")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Warning: failed to load SN data ({exc})")
            datasets['sn'] = None
    else:
        datasets['sn'] = None

    if args.use_cmb:
        datasets['cmb'] = CMBData()
        print("  CMB: constraints loaded")
    else:
        datasets['cmb'] = None

    if datasets['bao'] is not None and datasets['bao'].count_observables() > 0:
        fid_lcdm = LCDMModel()
        fid_params = [0.67, 0.31]
        print("\nFiducial BAO sanity check (ΛCDM: h=0.67, Ωm=0.31):")
        fid_max_pull = datasets['bao'].print_residual_table(fid_lcdm, fid_params, rd_value=147.0)
        print(f"  => Fiducial maximum |pull| = {fid_max_pull:.3f}")

        print("Fiducial fractional differences (|Δ|/obs):")
        max_frac = 0.0
        for point in datasets['bao'].data:
            obs_vec, theory_vec, _ = datasets['bao']._collect_observables(point, fid_lcdm, fid_params, 147.0)
            if obs_vec is None or obs_vec.size == 0:
                continue
            frac_diff = np.abs(obs_vec - theory_vec) / obs_vec
            if frac_diff.size:
                max_frac = max(max_frac, float(np.max(frac_diff)))
            if np.any(frac_diff > 0.01):
                print(f"  Warning: {point.get('name', 'BAO_point')} exceeds 1%: {frac_diff}")
        if max_frac <= 0.01:
            print("  All fiducial predictions agree within 1% of observations")

        if args.diagnose_lya_dh:
            diagnostic_flag = False
            with datasets['bao'].temporarily_drop('DESI Lyα GCcomb', 'DH_over_rd') as dropped:
                diagnostic_flag = dropped
                if dropped:
                    print("\nDiagnostic BAO residuals without Lyα DH/rd:")
                    diag_pull = datasets['bao'].print_residual_table(
                        fid_lcdm, fid_params, rd_value=147.0,
                        title='(Lyα DH/rd temporarily removed)'
                    )
                    print(f"  => Diagnostic maximum |pull| (Lyα DH removed) = {diag_pull:.3f}")
                else:
                    print("\nDiagnostic request: Lyα DH/rd observable unavailable for removal.")
            if diagnostic_flag:
                print("  Lyα DH/rd observable restored for covariance-weighted fitting.")

    if args.drop_lya_dh:
        if datasets['bao'] is not None:
            dropped_final = datasets['bao'].drop_observable('DESI Lyα GCcomb', 'DH_over_rd')
            if dropped_final:
                print("\n  BAO: Lyα DH/rd observable removed from the final fit")
            else:
                print("\n  Warning: Lyα DH/rd observable not found or already removed")
        else:
            print("\n  Warning: cannot drop Lyα DH/rd because BAO data are not loaded")

    final_bao_count = datasets['bao'].count_observables() if datasets['bao'] is not None else 0
    print(f"  BAO (final): {final_bao_count} observables from {len(datasets['bao'].data) if datasets['bao'] else 0} entries in use")
    final_cov_entries = datasets['bao'].covariance_entry_count() if datasets['bao'] is not None else 0
    if final_cov_entries:
        print(f"    Covariance applied to {final_cov_entries} BAO entries")

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
        print(f"\n{'='*50}")
        print(f"Fitting {model.name} model...")
        print(f"Parameters: {model.param_names}")
        print(f"Bounds: {model.bounds}")

        obj = partial(total_chi2,
                     datasets=datasets,
                     model=model,
                     rd_mode=args.rd_mode)

        try:
            de_res = differential_evolution(
                obj,
                bounds=model.bounds,
                workers=args.workers,
                updating='deferred',
                polish=True,
                disp=True,
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
                print("[WARN] Skipping covariance estimation because the optimizer did not converge.")
            elif dof_candidate <= 0:
                print("[WARN] Skipping covariance estimation because degrees of freedom are non-positive.")
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
                    print(f"[WARN] covariance estimation failed: {exc}")
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"[WARN] covariance estimation raised an unexpected error: {exc}")

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

            print(f"\nOptimization {'succeeded' if de_res.success else 'failed'}")
            print(f"Message: {de_res.message}")
            print(f"Function evaluations: {de_res.nfev}")
            print(f"\nBest-fit parameters:")
            for name, value in zip(model.param_names, de_res.x):
                print(f"  {name:8s} = {value:.6f}")

            for (name, value), (low, high) in zip(zip(model.param_names, de_res.x), model.bounds):
                span = high - low
                if span <= 0:
                    continue
                lower_frac = (value - low) / span
                upper_frac = (high - value) / span
                if lower_frac < 0.05 or upper_frac < 0.05:
                    print(f"  Warning: parameter '{name}' is within 5% of its bounds ({low}, {high})")

            print(f"\nStatistics:")
            chi2_disp = stats.get('chi2')
            chi2_red_disp = stats.get('chi2_red')
            aic_disp = stats.get('aic')
            bic_disp = stats.get('bic')

            chi2_text = f"{chi2_disp:.3f}" if chi2_disp is not None else "undefined"
            chi2_red_text = f"{chi2_red_disp:.3f}" if chi2_red_disp is not None else "undefined"
            aic_text = f"{aic_disp:.3f}" if aic_disp is not None else "undefined"
            bic_text = f"{bic_disp:.3f}" if bic_disp is not None else "undefined"

            print(f"  χ² = {chi2_text}")
            print(f"  dof = {stats['dof']}")
            print(f"  χ²/dof = {chi2_red_text}")
            print(f"  AIC = {aic_text}")
            print(f"  BIC = {bic_text}")

            if datasets['bao'] is not None and datasets['bao'].count_observables() > 0:
                print(f"BAO residuals at best-fit {model.name}:")
                rd_for_residuals = 147.0
                if args.rd_mode == 'fit' and 'rd' in model.param_names:
                    rd_index = model.param_names.index('rd')
                    rd_candidate = de_res.x[rd_index]
                    if np.isfinite(rd_candidate) and rd_candidate > 0:
                        rd_for_residuals = rd_candidate
                best_pull = datasets['bao'].print_residual_table(model, de_res.x, rd_value=rd_for_residuals)
                print(f"  => Maximum |pull| at best-fit {model.name}: {best_pull:.3f}")

        except Exception as exc:  # pylint: disable=broad-except
            print(f"ERROR during optimization: {exc}")
            results[model.name] = {
                'success': False,
                'error': str(exc)
            }

    if 'LCDM' in results and '7D' in results:
        if results['LCDM']['success'] and results['7D']['success']:
            print(f"\n{'='*50}")
            print("Model Comparison:")
            print(f"{'Model':<10} {'χ²':<10} {'AIC':<10} {'BIC':<10}")
            print("-" * 40)
            for model_name in ['LCDM', '7D']:
                stats = results[model_name]['statistics']
                print(f"{model_name:<10} {stats['chi2']:<10.3f} "
                      f"{stats['aic']:<10.3f} {stats['bic']:<10.3f}")

            delta_aic = results['7D']['statistics']['aic'] - results['LCDM']['statistics']['aic']
            delta_bic = results['7D']['statistics']['bic'] - results['LCDM']['statistics']['bic']

            print(f"\nΔAIC (7D - ΛCDM) = {delta_aic:.3f}")
            print(f"ΔBIC (7D - ΛCDM) = {delta_bic:.3f}")

            if delta_aic < -2:
                print("AIC strongly prefers 7D model")
            elif delta_aic < 0:
                print("AIC slightly prefers 7D model")
            elif delta_aic < 2:
                print("AIC slightly prefers ΛCDM model")
            else:
                print("AIC strongly prefers ΛCDM model")

    try:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
        print(f"\nResults saved to {args.output}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: Could not save results: {exc}")

    return results


def cli():
    if sys.platform.startswith('win'):
        mp.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nFATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import sys

    cli()
