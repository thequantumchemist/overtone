from abc import ABC, abstractmethod
from typing import Callable
from ase import Atoms

class BaseEngine(ABC):

    @abstractmethod
    def calculate_trajectory(self, trajectory: list[Atoms], setup_calculator: Callable, **kwargs) -> list[Atoms]:
        """
        Calculate properties for a trajectory of Atoms objects using a provided calculator setup function.

        Parameters
        ----------
        trajectory : list[Atoms]
            A list of Atoms objects representing the trajectory to be calculated.
        setup_calculator : Callable
            A function that sets up and returns a calculator for the Atoms objects.
        properties : sequence of str, optional (keyword)
            ASE properties to evaluate, default ("energy",). "dipole" is only
            requested when IR intensities are wanted.
        Returns
        -------
        list[Atoms]
            A list of Atoms objects with calculated properties.
        """
        ...
    