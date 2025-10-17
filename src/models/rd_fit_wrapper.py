"""Wrapper model that introduces a free sound horizon parameter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RDFitWrapper:
    """Wrapper that augments a cosmology model with a free r_d parameter."""

    base_model: object
    rd_bounds: tuple[float, float] = (120.0, 170.0)

    def __post_init__(self):
        self.param_names = tuple(self.base_model.param_names) + ("rd",)
        self.bounds = tuple(self.base_model.bounds) + (self.rd_bounds,)
        self.name = f"{self.base_model.name}+rd"

    def __getattr__(self, item):
        if item.startswith('__') or item.startswith('_'):
            raise AttributeError(item)

        base_model = object.__getattribute__(self, 'base_model')
        if hasattr(base_model, item):
            return getattr(base_model, item)

        raise AttributeError(item)

    def __getstate__(self):
        return {
            'base_model': self.base_model,
            'param_names': self.param_names,
            'bounds': self.bounds,
            'name': self.name,
        }

    def __setstate__(self, state):
        self.base_model = state['base_model']
        self.param_names = state['param_names']
        self.bounds = state['bounds']
        self.name = state['name']

    def _split(self, params: Sequence[float]):
        core_len = len(self.base_model.param_names)
        core = params[:core_len]
        rd = params[core_len]
        return core, rd

    def E(self, z, params):
        core, _ = self._split(params)
        return self.base_model.E(z, core)

    def Hz(self, z, params):
        core, _ = self._split(params)
        return self.base_model.Hz(z, core)

    def regularization(self, params, datasets):
        if hasattr(self.base_model, 'regularization'):
            core, _ = self._split(params)
            return self.base_model.regularization(core, datasets)
        return 0.0

    def omega_b_h2(self, params):
        if hasattr(self.base_model, 'omega_b_h2'):
            core, _ = self._split(params)
            return self.base_model.omega_b_h2(core)
        return 0.02237
