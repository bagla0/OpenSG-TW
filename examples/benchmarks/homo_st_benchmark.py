"""
Homogenization stiffness benchmark — JAX-TW vs FEniCS solid, station 15.

Compares the 6x6 Timoshenko (and 4x4 Euler-Bernoulli) beam stiffness from:
  * MSG-TW (JAX)     : 1Dshell_15.yaml      (timoshenko_from_yaml)
  * FEniCS solid     : 2Dsolid_VABS_15.yaml (compute_timo_boun, run in WSL)

The FEniCS solid 6x6 is read from outputs/fenics_st15_timo.txt, written by the
WSL driver fenics_st15.py.  Both 6x6 use the order
[EA, GA12, GA13, GJ, EI2, EI3] (extension, two shears, torsion, two bendings).

Usage
-----
    # 1) JAX + comparison (this script)
    python examples/homo_st_benchmark.py
"""
import os
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
np.set_printoptions(precision=4, linewidth=140, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml

SHELL15 = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
LBL = ["EA  ", "GA12", "GA13", "GJ  ", "EI2 ", "EI3 "]


def main():
    print("=" * 70)
    print("Homogenization stiffness benchmark — station 15")
    print("=" * 70)
    EB, Timo, _ = timoshenko_from_yaml(SHELL15)
    print(f"JAX-TW  1Dshell_15.yaml\n  Timoshenko 6x6 diag [EA,GA12,GA13,GJ,EI2,EI3]:")
    for i, lb in enumerate(LBL):
        print(f"    {lb} = {Timo[i, i]:.6e}")

    fe_path = os.path.join(OUT, "fenics_st15_timo.txt")
    if not os.path.exists(fe_path):
        print(f"\n[!] FEniCS solid 6x6 not found at {fe_path}")
        print("    Run the WSL driver first:  bash run_fenics_st15.sh")
        return
    FE = np.loadtxt(fe_path)
    print(f"\nFEniCS solid  2Dsolid_VABS_15.yaml  (compute_timo_boun)\n"
          f"  6x6 diag:")
    for i, lb in enumerate(LBL):
        print(f"    {lb} = {FE[i, i]:.6e}")

    print("\nDiagonal comparison  [100*(JAX-FE)/FE]:")
    print(f"  {'term':5s} {'JAX-TW':>13s} {'FEniCS-solid':>14s} {'%diff':>9s}")
    for i, lb in enumerate(LBL):
        j, f = Timo[i, i], FE[i, i]
        print(f"  {lb} {j:13.5e} {f:14.5e} {(j-f)/f*100:8.2f}%")

    # full-matrix relative difference (Frobenius, normalized)
    rel = np.linalg.norm(Timo - FE) / np.linalg.norm(FE) * 100
    print(f"\n  Full 6x6 relative difference (Frobenius): {rel:.2f}%")
    print("\n  JAX-TW 6x6:\n", np.array(Timo))
    print("\n  FEniCS solid 6x6:\n", FE)


if __name__ == "__main__":
    main()
