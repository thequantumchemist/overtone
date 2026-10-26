# H₂ with CCSD and ADAPT-VQE

Anharmonic stretching frequency of H₂ in the STO-3G basis, computed with
OVERTONE from CCSD (PySCF) and from ADAPT-VQE on a simulated quantum computer
(Qiskit Aer statevector, UCCSD operator pool, SLSQP optimizer).

For two electrons in STO-3G, CCSD is exact (equal to full CI), and so is a
converged ADAPT-VQE. Any difference between the two methods therefore shows
how well ADAPT-VQE is converged and how strongly the finite differences
amplify the remaining error.

## Installation

The Qiskit stack needs its own environment (see the
[ase_quantum_vqe](https://github.com/thequantumchemist/ase_quantum_vqe) README):

```
conda create -n overtone-qc python=3.11
conda activate overtone-qc
conda install -c conda-forge pyscf
pip install git+https://github.com/thequantumchemist/ase_quantum_vqe.git
pip install overtone-vpt2
```

## Running

```
python run_h2.py            # default step 0.2
python run_h2.py --scan     # steps 0.1, 0.2, 0.3 and 0.4
python run_h2.py --n-jobs 6 # parallel numerical forces
```

The full scan takes about 6 minutes on one core. Results are written to
`results/h2_summary.csv`, and every run keeps its OVERTONE JSON files in
`results/<method>_step<step>/`.

## Settings

| Method | Calculator | Convergence |
|---|---|---|
| CCSD | `PySCFCalculator(basis="sto-3g", method="ccsd")` | SCF 1e-12 Eh, CCSD 1e-12 Eh (set in the script) |
| ADAPT-VQE (default) | `QiskitVQECalculator(basis="sto3g", backend="aer", maxiter=500)` | Qiskit SLSQP default, ftol = 1e-6 |
| ADAPT-VQE (tight) | same, with `optimizer=SLSQP(maxiter=500, ftol=1e-12)`, `vqe_eigenvalue=1e-12` | ftol = 1e-12 |

All runs use the energy route (both calculators compute forces numerically),
stencil 1, and the step given on the command line. The geometry is
optimised with each method separately.
