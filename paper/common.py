"""
Shared helpers for the paper scripts.

Every script accepts
    --test        replace ORCA by GFN2-xTB (tblite) for a quick local check
    --nprocs N    cores for ORCA (default 16)
    --shephex     distribute the single points with Shephex/SLURM (see engine())

The ORCA executables are taken from the environment variables ORCA_COMMAND
(ORCA 5.0.4, water) and ORCA6_COMMAND (ORCA 6.0.1, methane), with the cluster
paths of the original calculations as defaults.
"""

# Version of this helper module. The paper scripts check it at start-up, so an
# outdated copy of common.py gives a clear message instead of a traceback.
COMMON_VERSION = 3

import argparse
import csv
import os
import re
import warnings
from pathlib import Path

import numpy as np
from ase.optimize import BFGS
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=DeprecationWarning)

ORCA5 = os.environ.get("ORCA_COMMAND", "/cluster/applications/ORCA/orca_5_0_4/orca")
ORCA6 = os.environ.get(
    "ORCA6_COMMAND",
    "/cluster/applications/ORCA/orca_6_0_1_linux_x86-64_shared_openmpi416_avx2/orca",
)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--test", action="store_true",
                   help="use GFN2-xTB instead of ORCA (quick local check)")
    p.add_argument("--nprocs", type=int, default=16, help="cores for ORCA")
    p.add_argument("--shephex", action="store_true",
                   help="distribute single points via Shephex (SLURM)")
    p.add_argument("--dry-run", action="store_true",
                   help="only print the number of single points")
    return p


def orca(simpleinput, blocks="", nprocs=16, command=ORCA5, label="orca"):
    """Factory for an ASE ORCA calculator (new instance on every call)."""
    from ase.calculators.orca import ORCA, OrcaProfile

    def make():
        return ORCA(
            label=label,
            profile=OrcaProfile(command=command),
            orcasimpleinput=simpleinput,
            orcablocks=f"%pal nprocs {nprocs} end\n{blocks}".strip(),
        )
    return make


def xtb():
    """GFN2-xTB stand-in for --test runs (tight SCF, see the SI)."""
    from tblite.ase import TBLite
    return lambda: TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)


def engine(use_shephex):
    """
    SerialEngine (default) runs all single points inside the current job.
    With --shephex the points are distributed as SLURM jobs. Adjust the
    SLURM options below to your cluster.
    """
    if not use_shephex:
        return None  # calculate_anharmonics then uses SerialEngine
    from shephex.executor import SlurmExecutor
    from overtone.engines import ShephexEngine
    return ShephexEngine(SlurmExecutor(
        partition=os.environ.get("SLURM_PARTITION", "standard"),
        ntasks=1, cpus_per_task=16, time="02:00:00",
    ))


def optimise(atoms, setup, fmax=1e-4, constraint=None, logfile="opt.log", save=None):
    """
    Tight geometry optimisation (VPT2 needs an accurate minimum).

    save : path of a file for the optimised structure. If it exists, the
        structure is read instead of optimised again. This makes restarts
        exact: the displaced structures of a restarted run then match the
        saved single points to the last digit.
    """
    from ase.io import read, write

    if save is not None and Path(save).exists():
        print(f"Restart: using the optimised structure in {save}")
        return read(save)
    atoms = atoms.copy()
    if constraint is not None:
        atoms.set_constraint(constraint)
    atoms.calc = setup()
    if logfile not in (None, "-"):
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    BFGS(atoms, logfile=logfile).run(fmax=fmax, steps=500)
    atoms.set_constraint()
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        clean = atoms.copy()
        write(save, clean)
    return atoms


def assign_modes(harmonic, reference):
    """
    Match computed modes to reference labels by their harmonic frequencies.

    Parameters
    ----------
    harmonic : {mode index: harmonic frequency}
    reference : {label: reference harmonic frequency}

    Returns
    -------
    {label: mode index}
    """
    modes, labels = list(harmonic), list(reference)
    cost = np.abs(np.subtract.outer([reference[l] for l in labels],
                                    [harmonic[m] for m in modes]))
    rows, cols = linear_sum_assignment(cost)
    return {labels[r]: modes[c] for r, c in zip(rows, cols)}


def fundamentals(result):
    """{mode index: VPT2Transition} for the fundamental bands."""
    return {next(iter(t.quanta)): t for t in result.transitions if t.kind == "fundamental"}


def band(result, quanta):
    """Transition with the given {mode: quanta}, or None."""
    for t in result.transitions:
        if t.quanta == quanta:
            return t
    return None


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {path}")


def write_latex(rows, columns, path, fmt="{:.1f}"):
    """Table body (rows only) for \\input in the manuscript or SI."""
    lines = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append(" & ".join(cells) + r" \\")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"written {path}")


def hitran_band_intensities(par_file, isotopologue=1, abundance=0.997317):
    """
    Sum HITRAN line intensities per vibrational band of H2(16)O starting
    from the ground state (000) and convert them to km/mol.

    par_file : HITRAN .par file (160-character format) downloaded from
        hitran.org, e.g. all H2O lines between 1000 and 8000 cm^-1.
    Returns {(v1, v2, v3): intensity in km/mol}
    """
    N_A = 6.02214076e23
    bands = {}
    with open(par_file) as f:
        for line in f:
            if len(line) < 100 or int(line[2]) != isotopologue:
                continue
            # vibrational labels, e.g. "0 1 0". Unassigned states are marked
            # with -2 (e.g. "-2 1-2"); they carry a negligible intensity and
            # are skipped.
            upper = tuple(int(x) for x in re.findall(r"-?\d+", line[67:82])[-3:])
            lower = tuple(int(x) for x in re.findall(r"-?\d+", line[82:97])[-3:])
            if len(upper) != 3 or min(upper) < 0 or lower != (0, 0, 0):
                continue
            S = float(line[15:25])  # cm/molecule at 296 K, abundance-scaled
            bands[upper] = bands.get(upper, 0.0) + S
    return {k: v / abundance * N_A * 1e-5 for k, v in bands.items()}


def quanta_of(label, modes):
    """'v1+v3' or '2v2' -> {mode index: quanta}, with modes = {'v1': index, ...}."""
    q = {}
    for part in label.split("+"):
        n = int(part[0]) if part[0].isdigit() else 1
        name = part[1:] if part[0].isdigit() else part
        q[modes[name]] = q.get(modes[name], 0) + n
    return q


def water_modes(result):
    """Water mode labels by harmonic frequency: bend < sym. stretch < antisym. stretch."""
    f = fundamentals(result)
    order = sorted(f, key=lambda m: f[m].harmonic)
    return {"v2": order[0], "v1": order[1], "v3": order[2]}


def require(version, script):
    """Stop with a clear message if this common.py is older than the script."""
    if COMMON_VERSION < version:
        raise SystemExit(
            f"{script} needs paper/common.py version {version} or newer, found "
            f"{COMMON_VERSION}. Copy common.py from the same release as the script."
        )
