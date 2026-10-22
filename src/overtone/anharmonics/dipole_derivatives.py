"""
Dipole derivatives along dimensionless normal coordinates (for IR intensities).

The derivatives are taken by nested finite differences on the same grid of
displacement keys as the force field, so in practice the dipoles can be read
from the anharmonic trajectory that was already computed. They are only
needed (and only evaluated) when intensities are requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer

from overtone.utils.numerical_derivatives import NumericalDerivativesCalculator


class DipoleDerivativesResult(BaseModel):
    """
    Attributes
    ----------
    mu1 : np.ndarray
        d mu / d q_i, shape (n_modes, 3), e*Angstrom
    mu2 : np.ndarray
        d^2 mu / d q_i d q_j, shape (n_modes, n_modes, 3), e*Angstrom
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mu1: np.ndarray
    mu2: np.ndarray

    @field_serializer("mu1", "mu2")
    def _ser(self, value: np.ndarray):
        return value.tolist()

    def dump(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DipoleDerivativesResult":
        with open(filepath) as f:
            data = json.load(f)
        return cls(mu1=np.array(data["mu1"]), mu2=np.array(data["mu2"]))


class DipoleDerivativesCalculator:
    """
    Parameters
    ----------
    n_modes : int
        Total number of modes (length of the displacement keys)
    active_modes : Sequence[int]
    delta : float
        Step in dimensionless normal coordinates (same as the force field)
    stencil : int
        Finite-difference stencil (same meaning as for the force field)
    """

    def __init__(self, n_modes: int, active_modes: Sequence[int], delta: float, stencil: int = 1):
        self.n_modes = n_modes
        self.active_modes = list(active_modes)
        self.calc = NumericalDerivativesCalculator(n_modes, delta, stencil)

    def _index_lists(self) -> List[List[int]]:
        act = self.active_modes
        lists = [[i] for i in act]
        lists += [[i, j] for a, i in enumerate(act) for j in act[a:]]
        return lists

    def get_displacement_keys(self) -> np.ndarray:
        return self.calc.keys_for(self._index_lists())

    def calculate(self, dipoles: Dict[Tuple[int, ...], np.ndarray]) -> DipoleDerivativesResult:
        """
        Parameters
        ----------
        dipoles : {displacement key: dipole vector in e*Angstrom}
        """
        missing = [tuple(k) for k in self.get_displacement_keys() if tuple(k) not in dipoles]
        if missing:
            raise KeyError(f"{len(missing)} displacement keys have no dipole, e.g. {missing[0]}")
        n = self.n_modes
        mu1 = np.zeros((n, 3))
        mu2 = np.zeros((n, n, 3))
        for i in self.active_modes:
            mu1[i] = self.calc.derivative([i], dipoles)
        act = self.active_modes
        for a, i in enumerate(act):
            for j in act[a:]:
                mu2[i, j] = mu2[j, i] = self.calc.derivative([i, j], dipoles)
        return DipoleDerivativesResult(mu1=mu1, mu2=mu2)
