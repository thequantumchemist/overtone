# 7. Running on a cluster

The single points of a VPT2 calculation are independent of each other, so
they can run in parallel on a cluster. This tutorial repeats the
[quickstart](01-quickstart.md) calculation of water, with every single point
submitted as a Slurm job through [shephex](https://pypi.org/project/shephex/).

This tutorial demonstrates:

- Submitting the single points to Slurm with `ShephexEngine` and a `SlurmExecutor`
- Running the workflow in stages by rerunning the script
- What happens when single points fail, and how to rerun them

GFN2-xTB on water is far too cheap to need a cluster; it stands in for an
expensive calculator such as DFT.

```bash
pip install "overtone-vpt2[tutorial,cluster]"
```

## Run the script until it finishes

The script does not wait for the jobs. Waiting would keep a Python process
running on the login node for as long as the jobs take, possibly days.
Instead, every run submits what can be computed now and stops:

```bash
export OMP_NUM_THREADS=1    # passed on to the jobs
python 07_cluster.py        # submits the 19 harmonic single points
python 07_cluster.py        # when finished: submits the 181 anharmonic ones
python 07_cluster.py        # when finished: VPT2 and the spectrum
```

The anharmonic displacements follow from the harmonic normal modes, so the
two stages cannot be submitted together. Running the script while jobs are
still queued is harmless: nothing is submitted twice, and the script reports
how many single points are finished.

## Step 1: The executor

```python
--8<-- "tutorials/07_cluster.py:executor"
```

The keyword arguments of `SlurmExecutor` become `#SBATCH` options, so any
sbatch option, such as `partition`, `mem` or `account`, can be given here.
`time` is per single point: each one runs as a task of a job array.
`environment="current"` runs the jobs in the Python environment the script
was started from, which must have OVERTONE and the calculator installed.

Shephex only submits from the script that is run, so the calculation is
inside an `if __name__ == "__main__":` block.

## Step 2: The workflow

```python
--8<-- "tutorials/07_cluster.py:workflow"
```

`calculate_anharmonics` is called exactly as in the quickstart, with the
engine added. While single points are queued or running it raises
`CalculationsPending`, which the script catches and prints, for example

```text
0/19 single points in water_cluster/harmonics are finished (19 submitted).
Rerun when the jobs have finished; finished single points are not repeated. ...
```

Once everything is finished, the same call returns the result. Every
single point is stored in its own directory under `water_cluster/harmonics/`
and `water_cluster/anharmonics/`, and finished ones are read back instead of
being repeated. To see how many single points a calculation needs before
submitting anything, run it with `dry_run=True`.

The progress of the jobs can be followed with shephex:

```bash
hex report water_cluster/anharmonics
```

## When single points fail

If any single point fails, for example because the SCF does not converge,
the run stops with `CalculationsFailed`. The error lists the failed
structures and the first error message. The full output of each job, the
`slurm-*.out` file, is moved to the directory of its single point, which
`hex report` lists.

Failed single points are not rerun automatically, since rerunning them
unchanged usually fails again. After fixing the cause, for example with
tighter SCF settings in `setup_calculator`, rerun them with

```python
engine = ShephexEngine(executor, retry_failed=True)
```

This removes the failed single points and submits them again with the new
calculator. Finished ones are kept.

A job that Slurm itself stops, because it exceeded its time or memory
limit, cannot report back. Its single point remains `submitted` or
`running`, and the script keeps reporting it as pending. If the script
reports pending single points while `squeue` shows no jobs, check
`hex report` and the `slurm-*.out` file of the single point.

## Output files

- `water_cluster/`: one directory per single point, with its `calc_<i>.traj` and Slurm output
- `slurm/`: the job scripts
- `water_cluster.png`: the spectrum, after the last run

## Full script

```python title="07_cluster.py"
--8<-- "tutorials/07_cluster.py"
```
