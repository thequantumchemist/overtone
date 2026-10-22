from typing import Callable, Sequence

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from overtone.engines.base_engine import BaseEngine


class SerialEngine(BaseEngine):
    """
    Runs all single points one after another in the current process.

    After each structure is calculated, its results are frozen in a
    SinglePointCalculator. This matters for calculators that are shared
    between structures, e.g. one on-the-fly learning FALCON instance: ASE
    caches results per calculator, so reading an earlier structure again
    would otherwise trigger a new calculation.
    """

    def calculate_trajectory(
        self,
        trajectory: list[Atoms],
        setup_calculator: Callable,
        properties: Sequence[str] = ("energy",),
        **kwargs,
    ) -> list[Atoms]:
        calculated_trajectory = []
        for atoms in trajectory:
            calc = setup_calculator()
            atoms.calc = calc
            results = {}
            for prop in properties:
                results[prop] = calc.get_property(prop, atoms)
            atoms.calc = SinglePointCalculator(atoms, **results)
            calculated_trajectory.append(atoms)
        return calculated_trajectory
