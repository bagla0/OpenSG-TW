# Installation

OpenSG-TW is a JAX package; the heavy linear algebra goes through Intel MKL **PARDISO**
(`pypardiso`), and `basix` tabulates the element basis/quadrature. No FEniCSx or MPI is required.

## Environment

```{tip}
On Windows the conda environment must be **prepended to `PATH`** (not merely activated) so the MKL /
PARDISO DLLs resolve. The reference environment used throughout these docs is `opensg_2_0_env`
(Python 3.12).
```

```bash
conda create -n opensg_tw python=3.12
conda activate opensg_tw
pip install "jax[cpu]" pypardiso fenics-basix numpy scipy pyyaml numba matplotlib
```

PowerShell (Windows) — prepend the env to `PATH` before running:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "C:\conda_envs\opensg_2_0_env;C:\conda_envs\opensg_2_0_env\Library\bin;C:\conda_envs\opensg_2_0_env\Scripts;" + $env:PATH
```

## Get the code

```bash
git clone https://github.com/bagla0/OpenSG-TW.git
cd OpenSG-TW
```

The package lives in `opensg_jax/fe_jax/` and mirrors the [OpenSG_2.0](https://github.com/KeithBallard/fea-in-jax)
JAX FEA architecture (basis/quadrature, sparse assembly, periodic dof-maps, PARDISO solve).

## Quick check — a 6×6 in three lines

The 2-D solid driver takes a 2-D solid SG YAML and returns the Timoshenko 6×6:

```python
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
C6 = compute_timo_from_yaml("prevabs_mh104/2Dsolid_VABS_mh_104.yaml")
print(C6)   # [EA, GA2, GA3, GJ, EI2, EI3] on the diagonal
```

Benchmark it against the VABS `.K`:

```powershell
python -m opensg_jax.fe_jax.benchmark_vabs `
  prevabs_mh104\2Dsolid_VABS_mh_104.yaml `
  "training data\opensg-FEniCS\data\mh104_training\mh104.sg.K"
```

For the shell models see the [tutorials](tutorials/index.md). The numbered runnable scripts under
`examples/` (`1_get_beam_props_*`) reproduce each tutorial from the command line.
