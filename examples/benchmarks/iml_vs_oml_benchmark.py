"""
OML vs IML reference homogenization — MSG-TW (JAX), station 15.

The 1Dshell YAML places the reference curve on the outer mold line (OML); the
thick spar-cap / web laminates then extend inward and OVERLAP at junctions,
double-counting material.  Referencing the inner mold line (IML) — nodes offset
inward by the local laminate thickness, plate reference flipped to the IML —
moves that material outward and reduces the overlap.  This compares the two
homogenized 6x6 stiffnesses (and the FEniCS solid, if available) so we can see
the effect before moving on to dehomogenization.

The IML mesh reuses the same connectivity (no new nodes/elements), so it stays
memory-light.
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

SHELL = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
LBL = ["EA  ", "GA12", "GA13", "GJ  ", "EI2 ", "EI3 "]


def main(yaml_path=SHELL):
    print("=" * 86)
    print(f"Reference-surface homogenization: OML / CENTROID / IML  |  "
          f"{os.path.basename(yaml_path)}")
    print("=" * 86)
    _, oml, _ = timoshenko_from_yaml(yaml_path, reference="OML")
    _, mid, _ = timoshenko_from_yaml(yaml_path, reference="CENTROID")
    _, iml, _ = timoshenko_from_yaml(yaml_path, reference="IML")

    fe_path = os.path.join(OUT, "fenics_st15_timo.txt")
    FE = np.loadtxt(fe_path) if os.path.exists(fe_path) else None

    if FE is not None:
        print(f"  {'term':5s} {'OML':>12s} {'CENTROID':>12s} {'IML':>12s} "
              f"{'FEsolid':>12s} | {'OML%':>7s} {'MID%':>7s} {'IML%':>7s}  (vs FE)")
        for i, lb in enumerate(LBL):
            o, c, m, f = oml[i, i], mid[i, i], iml[i, i], FE[i, i]
            print(f"  {lb} {o:12.4e} {c:12.4e} {m:12.4e} {f:12.4e} | "
                  f"{(o-f)/f*100:7.2f} {(c-f)/f*100:7.2f} {(m-f)/f*100:7.2f}")
        for tag, M in [("OML", oml), ("CENTROID", mid), ("IML", iml)]:
            print(f"  full-6x6 ||{tag:8s}-FE||/||FE|| = "
                  f"{np.linalg.norm(M-FE)/np.linalg.norm(FE)*100:.2f}%")
    else:
        print(f"  {'term':5s} {'OML':>12s} {'CENTROID':>12s} {'IML':>12s}")
        for i, lb in enumerate(LBL):
            print(f"  {lb} {oml[i,i]:12.4e} {mid[i,i]:12.4e} {iml[i,i]:12.4e}")
        print("  (run fenics_st15.py in WSL for the FEniCS-solid comparison)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SHELL)
