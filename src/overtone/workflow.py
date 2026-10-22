"""
One-call driver: harmonic analysis -> quartic force field -> VPT2 -> spectrum.

For full control, the same steps can be run one by one (see
tutorials/02_step_by_step.py).
"""

from pathlib import Path
from typing import Callable, List, Literal, Optional

import numpy as np
from ase import Atoms
from ase.io import write

from overtone.anharmonics import (
    AnharmonicDerivativesConfig,
    AnharmonicDerivativesSolver,
    VPT2Config,
    VPT2Result,
    VPT2Solver,
)
from overtone.engines import BaseEngine, SerialEngine
from overtone.harmonics import Harmonics
from overtone.harmonics.harmonics import is_linear_molecule
from overtone.utils.numerical_derivatives import (
    CartesianForceDerivativesCalculator,
    NumericalDerivativesCalculator,
)


def estimate_cost(
    atoms: Atoms,
    derivative_source: Literal["energy", "forces"] = "energy",
    stencil: int = 1,
    coupling: int = 3,
    active_modes: Optional[List[int]] = None,
    constrain_index: Optional[List[int]] = None,
    linear: Optional[bool] = None,
) -> dict:
    """
    Number of single-point calculations a run will need, without running it.

    Returns
    -------
    dict with 'harmonic', 'anharmonic', 'total' and 'n_modes'
    """
    n_eff = len(atoms) - (len(constrain_index) if constrain_index else 0)
    n_dof = 3 * n_eff
    if active_modes is None:
        if constrain_index:
            n_tr = 0
        else:
            lin = is_linear_molecule(atoms) if linear is None else linear
            n_tr = 5 if lin else 6
        active_modes = list(range(n_tr, n_dof))
    if derivative_source == "energy":
        calc = NumericalDerivativesCalculator(n_dof, 0.1, stencil)
    else:
        calc = CartesianForceDerivativesCalculator(
            n_dof, 0.1, np.eye(n_dof), np.ones(n_dof), None, stencil
        )
    calc.setup_derivative_indices(active_modes, [3, 4], coupling)
    n_harm = 1 + 6 * n_eff
    n_anh = len(calc.get_displacement_keys())
    return {
        "harmonic": n_harm,
        "anharmonic": n_anh,
        "total": n_harm + n_anh,
        "n_modes": len(active_modes),
    }


def calculate_anharmonics(
    atoms: Atoms,
    setup_calculator: Callable,
    directory: Path,
    engine: Optional[BaseEngine] = None,
    setup_harmonic_calculator: Optional[Callable] = None,
    # --- system -----------------------------------------------------------
    constrain_index: Optional[List[int]] = None,
    active_modes: Optional[List[int]] = None,
    exclude_below: float = 0.0,
    # --- harmonic analysis ------------------------------------------------
    harmonic_delta: float = 0.005,
    symmetrize_hessian: bool = True,
    project_translations_rotations: bool = True,
    linear: Optional[bool] = None,
    # --- quartic force field ----------------------------------------------
    derivative_source: Literal["energy", "forces"] = "energy",
    anharmonic_delta: float = 0.2,
    stencil: int = 1,
    coupling: int = 3,
    # --- VPT2 -------------------------------------------------------------
    method: Literal["standard", "split", "deperturbed", "dcpt2", "gvpt2"] = "gvpt2",
    coriolis: bool = True,
    resonance_window: float = 200.0,
    resonance_K1: float = 1.0,
    resonance_K2: float = 1.0,
    min_frequency: float = 20.0,
    # --- output -----------------------------------------------------------
    max_quanta: int = 2,
    intensities: bool = False,
    save_results: bool = False,
    dry_run: bool = False,
) -> Optional[VPT2Result]:
    """
    Anharmonic frequencies, overtones, combination bands and (optionally)
    IR intensities with VPT2.

    Parameters
    ----------
    atoms : ase.Atoms
        Equilibrium structure. Optimise it tightly with the same calculator
        (fmax ~ 1e-4 eV/A or lower). Finite differences amplify numerical
        noise, so use tight SCF convergence for all single points.
    setup_calculator : callable
        Function without arguments that returns a fresh ASE calculator.
    directory : Path
        Working directory; harmonics/ and anharmonics/ are created inside.
    setup_harmonic_calculator : callable, optional
        Separate calculator for the harmonic analysis (Hessian), e.g. a higher
        level of theory than the anharmonic part (hybrid scheme), a calculator
        with (numerical) gradients when the anharmonic part uses energies
        only, or the exact method when the anharmonic part uses FALCON.
        Default: setup_calculator.
    engine : BaseEngine, optional
        How the single points are run. SerialEngine (default) runs them one
        after another in this process. ShephexEngine distributes them over a
        queueing system (SLURM, ...) and can resume interrupted runs.

    System
    ------
    constrain_index : list of int, optional
        Atoms kept fixed (e.g. a surface slab). Their coordinates are not
        displaced, translations/rotations are not projected and Coriolis
        terms are switched off.
    active_modes : list of int, optional
        Mode indices to treat anharmonically. Default: all vibrational
        modes. Excluding very low-frequency or torsional modes reduces cost
        and avoids their (often unreliable) VPT2 treatment.
    exclude_below : float
        Leave out vibrational modes whose harmonic frequency (cm^-1) is
        below this value, decided after the harmonic analysis. Useful for
        torsions/free rotors whose index is not known beforehand, e.g.
        exclude_below=100. Imaginary modes are then excluded as well (with a
        notice). Ignored if active_modes is given. Default 0, which keeps all
        modes and stops for imaginary ones.

    Harmonic analysis
    -----------------
    harmonic_delta : float
        Cartesian step for the Hessian in Angstrom (default 0.005). Smaller
        steps reduce the truncation error from anharmonicity (important for
        torsions) but amplify numerical noise, so converge the SCF tightly
        (e.g. tblite accuracy=0.01, ORCA VeryTightSCF).
    symmetrize_hessian : bool
        Use (H + H^T)/2 (default True).
    project_translations_rotations : bool
        Eckart projection of the rigid-body motions (default True).
    linear : bool, optional
        Override the automatic linear-molecule detection.

    Quartic force field
    -------------------
    derivative_source : 'energy' or 'forces'
        Differentiate energies (any calculator) or analytic forces (needs
        fewer single points, recommended when forces are available).
    anharmonic_delta : float
        Step in dimensionless (reduced) normal coordinates q, not in
        Angstrom (default 0.2). The Cartesian displacement follows from the
        mode's frequency and the atomic masses (stiff modes and heavy atoms
        move less). Fourth derivatives amplify numerical noise as
        1/delta^4: with energies converged to 1e-8 Eh, delta = 0.1 scatters
        water fundamentals by ~10 cm^-1, delta = 0.2 by <1 cm^-1. Above
        ~0.3 the truncation error grows for floppy modes (torsions).
    stencil : int
        Finite-difference stencil: 0 (2-point), 1 (5-point, default),
        2 (7-point), 3 (9-point). Higher stencils need many more single
        points and rarely change the result (check with estimate_cost()).
    coupling : int
        3 (default): all cubic constants phi_ijk. 2: only phi_iij-type
        constants, cheaper but misses three-mode couplings (and therefore
        type-2 Fermi resonances).

    VPT2
    ----
    method : str
        'gvpt2' (default): deperturbed VPT2 plus variational treatment of
        every group of Fermi-coupled states (energies and intensities).
        Identical to standard VPT2 when no resonance is detected.
        'standard': textbook VPT2, no resonance treatment.
        'deperturbed': resonant terms removed (Martin et al. 1995), no
        variational step; resonant levels are deperturbed values.
        'dcpt2': degeneracy-corrected PT2 (Kuhler et al. 1996).
        'split': standard VPT2 reporting cubic and quartic contributions.
    coriolis : bool
        Include Coriolis contributions to X (default True).
    resonance_window, resonance_K1, resonance_K2 : float
        Fermi-resonance test (cm^-1): frequency window and the Martin
        thresholds for type 1 (2w_j ~ w_i) and type 2 (w_j + w_k ~ w_i).
        Used by 'deperturbed' and to flag resonant bands.
    min_frequency : float
        Active modes below this harmonic frequency (cm^-1), or imaginary
        ones, stop the run right after the harmonic analysis, before any
        expensive anharmonic single points (default 20). Modes below
        100 cm^-1 only trigger a warning.

    Output
    ------
    max_quanta : int
        Highest total number of quanta in the band list (2: overtones and
        binary combinations; 3: adds the second-overtone region).
    intensities : bool
        IR intensities (default False). Requests dipole moments for the
        displaced structures; keep False for energy-only calculators such
        as ML potentials.
    save_results : bool
        Write JSON files of every intermediate result to `directory`.
    dry_run : bool
        Only print and return the number of single points (no calculation).

    Returns
    -------
    VPT2Result (or the cost dictionary if dry_run=True)
    """
    if dry_run:
        cost = estimate_cost(
            atoms, derivative_source, stencil, coupling, active_modes,
            constrain_index, linear,
        )
        print(
            f"OVERTONE dry run: {cost['n_modes']} active modes, "
            f"{cost['harmonic']} harmonic + {cost['anharmonic']} anharmonic "
            f"= {cost['total']} single points"
            + (" (upper bound: modes below exclude_below are only removed after "
               "the harmonic analysis)" if exclude_below > 0 and active_modes is None else "")
        )
        return cost

    directory = Path(directory)
    harmonics_directory = directory / "harmonics"
    anharmonics_directory = directory / "anharmonics"
    for d in (directory, harmonics_directory, anharmonics_directory):
        d.mkdir(parents=True, exist_ok=True)
    harmonics_path = harmonics_directory / "harmonic_displacements.traj"
    anharmonics_path = anharmonics_directory / "anharmonic_displacements.traj"

    if engine is None:
        engine = SerialEngine()

    # ---------------- harmonic analysis ----------------
    harmonics = Harmonics(
        atoms,
        delta=harmonic_delta,
        constrain_index=constrain_index,
        symmetrize_hessian=symmetrize_hessian,
        project_translations_rotations=project_translations_rotations,
        linear=linear,
    )
    trajectory = engine.calculate_trajectory(
        harmonics.calculate_displacements(),
        setup_calculator=setup_harmonic_calculator or setup_calculator,
        trajectory_path=harmonics_path,
        output_directory=harmonics_directory,
        properties=["energy", "forces"],
    )
    harmonics_result = harmonics.calculate_hessian(trajectory)
    harmonics_result.summary()
    if save_results:
        harmonics_result.dump(harmonics_directory / "harmonics_result.json")
        write(harmonics_path, trajectory)

    if active_modes is None:
        if exclude_below > 0:
            # explicit request: also drops imaginary modes (e.g. the torsion of
            # a rotamer that is a maximum along a nearly free rotation)
            active_modes = [
                m for m in harmonics_result.vibrational_modes
                if harmonics_result.freq[m] >= exclude_below
                and harmonics_result.ifreq[m] < 0.05
            ]
        else:
            active_modes = list(harmonics_result.vibrational_modes)
        excluded = [m for m in harmonics_result.vibrational_modes if m not in active_modes]
        if excluded:
            print(f"Excluded modes below {exclude_below} cm^-1: " + ", ".join(
                f"{m} ({harmonics_result.ifreq[m]:.1f}i)" if harmonics_result.ifreq[m] >= 0.05
                else f"{m} ({harmonics_result.freq[m]:.1f})" for m in excluded))
    harmonics_result.check_active_modes(active_modes, min_frequency=min_frequency)

    # ---------------- quartic force field ----------------
    constrained = bool(constrain_index)
    unconstrained = harmonics.constraints.unconstrained_index
    config = AnharmonicDerivativesConfig.from_harmonics_result(
        harmonics_result=harmonics_result,
        atoms=atoms,
        active_modes=active_modes,
        delta=anharmonic_delta,
        coupling=coupling,
        stencil=stencil,
        constrained=constrained,
        unconstrained_index=unconstrained if constrained else None,
        force_indices=harmonics.constraints.force_indices if constrained else None,
    )
    if save_results:
        config.dump(anharmonics_directory / "derivatives_config.json")
    derivative_solver = AnharmonicDerivativesSolver(config=config)

    if derivative_source == "energy":
        trajectory, keys = derivative_solver.generate_displacement_trajectory()
        properties = ["energy"]
    elif derivative_source == "forces":
        trajectory, keys = derivative_solver.generate_displacement_trajectory_for_forces()
        properties = ["energy", "forces"]
    else:
        raise ValueError(
            f"derivative_source must be 'energy' or 'forces', got {derivative_source!r}"
        )
    if intensities:
        properties.append("dipole")  # only requested on demand

    trajectory = engine.calculate_trajectory(
        trajectory,
        setup_calculator=setup_calculator,
        trajectory_path=anharmonics_path,
        output_directory=anharmonics_directory,
        properties=properties,
    )
    if derivative_source == "energy":
        derivative_result = derivative_solver.calculate_derivatives(trajectory, keys)
    else:
        derivative_result = derivative_solver.calculate_derivatives_from_forces(trajectory)
    if save_results:
        derivative_result.dump(anharmonics_directory / "derivatives_result.json")
        write(anharmonics_path, trajectory)

    # ---------------- dipole derivatives (on request) ----------------
    dipole_result = None
    if intensities:
        missing = derivative_solver.missing_dipole_keys(keys)
        dip_traj, dip_keys = list(trajectory), np.asarray(keys)
        if len(missing):  # only for unusual coupling/stencil choices
            extra_dir = anharmonics_directory / "dipole_extra"
            extra_dir.mkdir(exist_ok=True)
            extra = engine.calculate_trajectory(
                derivative_solver.create_displaced_trajectory(missing),
                setup_calculator=setup_calculator,
                trajectory_path=extra_dir / "dipole_displacements.traj",
                output_directory=extra_dir,
                properties=["energy", "dipole"],
            )
            dip_traj += list(extra)
            dip_keys = np.vstack([dip_keys, missing])
        dipole_result = derivative_solver.calculate_dipole_derivatives(dip_traj, dip_keys)
        if save_results:
            dipole_result.dump(anharmonics_directory / "dipole_derivatives.json")

    # ---------------- VPT2 ----------------
    vpt2_config = VPT2Config.from_harmonics_result(
        harmonics_result=harmonics_result,
        active_modes=active_modes,
        atoms=None if constrained else atoms,  # no Coriolis for constrained systems
        coupling=coupling,
        method=method,
        max_quanta=max_quanta,
        coriolis=coriolis,
        resonance_window=resonance_window,
        resonance_K1=resonance_K1,
        resonance_K2=resonance_K2,
    )
    if save_results:
        vpt2_config.dump(anharmonics_directory / "vpt2_config.json")

    vpt2_result = VPT2Solver(
        config=vpt2_config,
        derivatives=derivative_result,
        dipole_derivatives=dipole_result,
    ).calculate()
    vpt2_result.summary(active_modes=active_modes)
    vpt2_result.transitions_table()
    if save_results:
        vpt2_result.dump(anharmonics_directory / "vpt2_result.json")
    return vpt2_result


def recompute_vpt2(
    directory,
    method: Literal["standard", "split", "deperturbed", "dcpt2", "gvpt2"] = "gvpt2",
    intensities: bool = True,
    **config_overrides,
) -> VPT2Result:
    """
    Repeat only the VPT2 step of a finished run (save_results=True), e.g.
    with another method, other resonance thresholds or more quanta. No new
    single points are calculated.

    Parameters
    ----------
    directory : run directory given to calculate_anharmonics
    method : VPT2 variant
    intensities : use saved dipole derivatives if present (default True)
    **config_overrides : any VPT2Config field, e.g. max_quanta=3, coriolis=False

    Example
    -------
    >>> deperturbed = recompute_vpt2("run", method="deperturbed")
    """
    from overtone.anharmonics import AnharmonicDerivativesResult
    from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult

    anh = Path(directory) / "anharmonics"
    saved = VPT2Config.load(anh / "vpt2_config.json")
    fields = {name: getattr(saved, name) for name in VPT2Config.model_fields}
    fields.update(method=method, **config_overrides)
    config = VPT2Config(**fields)  # re-validated, incl. method aliases
    derivatives = AnharmonicDerivativesResult.load(anh / "derivatives_result.json")
    dipoles = None
    if intensities and (anh / "dipole_derivatives.json").exists():
        dipoles = DipoleDerivativesResult.load(anh / "dipole_derivatives.json")
    return VPT2Solver(config, derivatives, dipole_derivatives=dipoles).calculate()

