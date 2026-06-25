"""
Kirchhoff (C1) thin-walled MSG shell -> Timoshenko 6x6 stiffness.

Reads the same 1D structure-genome strip_iso_1D.yaml and runs the Kirchhoff
MSG-TW solve.  Kirchhoff wall kinematics: cubic Hermite C1, 6 DOF/node
[w1,w1',w2,w2',w3,w3'] (value + arc-slope), NO transverse shear -- the wall
curvatures carry second contour derivatives of w, so the V1 (shear-warping)
condensation alone produces the beam transverse-shear GA.  Prints the 6x6 at the
centre and the OML references.  Order [ext, shear2, shear3, twist, bend2, bend3].

`solve_tw_from_yaml` runs the full Kirchhoff pipeline internally:
  per-ply ABD (1D SG, parallel-axis shifted to the reference) -> Hermite mesh
  (corner value+slope DOFs) -> assemble_system_matrices_hermite (Dhh,Dhe,Dee,
  Dll,Dhl,Dle by energy autodiff) -> build_constraints_hermite (C, Psi) ->
  solve_fluctuation_field (KKT -> V0, EB 4x4) -> prepare_v1_rhs -> PARDISO V1 ->
  finalize_v1_and_compute_deff (-> sorted Timoshenko 6x6).

Run (Windows):
  $env:PATH = "C:\\conda_envs\\opensg_2_0_env;...;" + $env:PATH   # see CLAUDE.md
  & "C:\\conda_envs\\opensg_2_0_env\\python.exe" strip_Kirchhoff.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

YAML = os.path.join(HERE, "strip_iso_1D.yaml")
LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]


def kirchhoff_timoshenko_6x6(yaml_path, frac):
    """Kirchhoff MSG-TW Timoshenko 6x6 at reference `frac` (0.0=OML, 0.5=centre)."""
    out = solve_tw_from_yaml(yaml_path, frac=frac)
    return np.asarray(out["Timo"])


def show(tag, C6):
    print(f"\n=== Kirchhoff Timoshenko 6x6  ({tag})  order {LBL} ===")
    for i in range(6):
        print("  " + "".join(f"{C6[i, j]:14.4e}" for j in range(6)))
    d = np.diag(C6)
    print("  diagonal:  EA={:.4e}  GA2={:.4e}  GA3={:.4e}  GJ={:.4e}  EI2={:.4e}  EI3={:.4e}"
          .format(d[0], d[1], d[2], d[3], d[4], d[5]))


if __name__ == "__main__":
    print(f"reading 1D genome: {YAML}")
    show("centre / mid-surface", kirchhoff_timoshenko_6x6(YAML, frac=0.5))
    show("OML", kirchhoff_timoshenko_6x6(YAML, frac=0.0))
