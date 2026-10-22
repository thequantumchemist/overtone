import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from overtone.engines.base_engine import BaseEngine


class SerialEngine(BaseEngine):
    """
    Runs all single points one after another in the current process.

    Restart: when an output directory is given (calculate_anharmonics always
    passes one), every finished structure is written immediately to
    ``<output_directory>/calculations/job_<index>.traj`` together with its
    results. A new run in the same directory reuses these files and only
    calculates the missing structures. A file is reused only if the atoms
    and positions match and all requested properties are present.

    After each structure, the results are frozen in a SinglePointCalculator.
    This matters for calculators that are shared between structures, e.g. an
    on-the-fly learning FALCON instance: ASE caches results per calculator,
    so reading an earlier structure again would otherwise trigger a new
    calculation.
    """

    def __init__(self, restart: bool = True, position_tolerance: float = 1e-8):
        """
        Parameters
        ----------
        restart : bool
            Write every finished structure to disk and reuse finished
            structures of an earlier run (default True).
        position_tolerance : float
            Largest position difference (Angstrom) for which a saved
            structure counts as the same structure (default 1e-8).
        """
        self.restart = restart
        self.position_tolerance = position_tolerance

    def _load(self, path: Path, atoms: Atoms, properties: Sequence[str]) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            saved = read(path)
        except Exception:
            return None  # e.g. a file truncated by a crash
        if saved.get_chemical_symbols() != atoms.get_chemical_symbols():
            return None
        if not np.allclose(saved.positions, atoms.positions, atol=self.position_tolerance, rtol=0):
            return None
        results = getattr(saved.calc, "results", {}) or {}
        if not all(p in results for p in properties):
            return None
        return {p: results[p] for p in properties}

    def calculate_trajectory(
        self,
        trajectory: list[Atoms],
        setup_calculator: Callable,
        properties: Sequence[str] = ("energy",),
        output_directory: Optional[Path | str] = None,
        **kwargs,
    ) -> list[Atoms]:
        cache = None
        if self.restart and output_directory is not None:
            cache = Path(output_directory) / "calculations"
            cache.mkdir(parents=True, exist_ok=True)

        calculated_trajectory, reused = [], 0
        n, start = len(trajectory), time.time()
        for i, atoms in enumerate(trajectory):
            path = cache / f"job_{i:06d}.traj" if cache is not None else None
            results = self._load(path, atoms, properties) if path is not None else None
            if results is not None:
                reused += 1
            else:
                calc = setup_calculator()
                atoms.calc = calc
                results = {prop: calc.get_property(prop, atoms) for prop in properties}
                # progress on stderr: visible in cluster logs even if stdout is redirected
                print(f"[OVERTONE] {i + 1}/{n} done ({reused} reused, "
                      f"{time.time() - start:.0f} s)", file=sys.stderr, flush=True)
            atoms.calc = SinglePointCalculator(atoms, **results)
            if path is not None and self._load(path, atoms, properties) is None:
                tmp = path.with_suffix(".tmp.traj")
                write(tmp, atoms)
                os.replace(tmp, path)  # atomic: no half-written files after a crash
            calculated_trajectory.append(atoms)

        if cache is not None and reused:
            print(f"Restart: reused {reused} of {len(trajectory)} finished structures "
                  f"from {cache}", flush=True)
            print(f"[OVERTONE] restart: reused {reused} of {n} structures from {cache}",
                  file=sys.stderr, flush=True)
        return calculated_trajectory
