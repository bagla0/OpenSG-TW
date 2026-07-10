"""1_run_solid_taper_jax.py -- JAX solid tapered-segment Timoshenko 6x6, validated
against FEniCS-OpenSG (opensg.core.solid.compute_stiffness(Taper=True)) on the SAME
segment YAML.

    python 1_run_solid_taper_jax.py <solid_segment.yaml> [--fenics]

Prints DOF / time / the three 6x6 (L boundary, R boundary, segment) from JAX, and with
--fenics also the FEniCS references + the %err 6x6 (cutoff |ref|>max/1e6 per the repo
convention)."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from opensg_jax.fe_jax.solid_taper import compute_timo_taper_solid

ORDER = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def show(title, S):
    print("\n%s" % title)
    for r in range(6):
        print("  " + "  ".join("% .5e" % S[r, c] for c in range(6)))
    print("  diagonal: " + "  ".join("%s=%.4e" % (ORDER[i], S[i, i]) for i in range(6)))


def pct_err(S, R):
    cut = np.abs(R).max() / 1e6
    E = np.full((6, 6), np.nan)
    m = np.abs(R) > cut
    E[m] = 100.0 * (S[m] - R[m]) / R[m]
    return E


def show_err(title, E):
    print("\n%s  (%% err vs FEniCS, blank = below cutoff)" % title)
    for r in range(6):
        print("  " + "  ".join(("%8.3f" % E[r, c]) if np.isfinite(E[r, c]) else "     .  "
                               for c in range(6)))
    print("  max |err| = %.3f %%" % np.nanmax(np.abs(E)))


def run_fenics(yaml_path):
    from opensg.mesh.segment import SolidSegmentMesh
    from opensg.core.solid import compute_stiffness
    yaml_abs = os.path.abspath(yaml_path)
    d = os.path.dirname(yaml_abs) or "."
    cwd = os.getcwd()
    os.chdir(d)                                            # gmsh scratch local
    try:
        sm = SolidSegmentMesh(yaml_abs)
        mp, _den = sm.material_database
        t0 = time.time()
        S, V0, V1s, DL, DR = compute_stiffness(mp, sm.meshdata, sm.left_submesh,
                                               sm.right_submesh, Taper=True)
        dt = time.time() - t0
    finally:
        os.chdir(cwd)
    sym = lambda M: 0.5 * (np.asarray(M) + np.asarray(M).T)
    return sym(DL), sym(DR), sym(S), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("--fenics", action="store_true", help="also run FEniCS + %%err")
    a = ap.parse_args()

    DL, DR, Dseg, info = compute_timo_taper_solid(a.yaml)
    print("\n=== JAX solid taper  [%s] ===" % os.path.basename(a.yaml))
    print("  elements=%d  dof=%d (free %d)  boundary %.2fs  segment %.2fs"
          % (info["nelem"], info["dof"], info["nfree"], info["t_boundary"], info["t_segment"]))
    show("JAX  L-boundary Timoshenko 6x6:", DL)
    show("JAX  R-boundary Timoshenko 6x6:", DR)
    show("JAX  SEGMENT Timoshenko 6x6:", Dseg)

    if a.fenics:
        FL, FR, FS, dt = run_fenics(a.yaml)
        print("\n=== FEniCS reference (compute_stiffness Taper=True, %.2fs) ===" % dt)
        show("FEniCS L-boundary:", FL)
        show("FEniCS R-boundary:", FR)
        show("FEniCS SEGMENT:", FS)
        show_err("L-boundary", pct_err(DL, FL))
        show_err("R-boundary", pct_err(DR, FR))
        show_err("SEGMENT", pct_err(0.5 * (Dseg + Dseg.T), FS))


if __name__ == "__main__":
    main()
