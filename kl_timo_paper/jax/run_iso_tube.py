"""JAX gradient-Kirchhoff (OpenSG-2.0) Timoshenko 6x6 for the report's isotropic
circular tube -- fills the JAX-KL column of the analytical-validation table.

Report Case-1 (Opensg_MSG.pdf, Table 3.1): isotropic tube, MEAN radius R = 5 m,
wall h = 0.2 m, E = 3.44 GPa, nu = 0.3, CENTER reference.  Reference values
(x10^6):  C11 21606 | C22=C33 4157 | C44 207650 | C55=C66 269680  (analytical),
VABS-solid 21622 / 4205 / 207299 / 269935.

Reuses the validated TW-paper tube infrastructure (gen_tube_yaml + tube_lib.homog,
which builds the C1-Hermite Kirchhoff 6x6 the same way solve_tw_from_yaml does).

Run (Windows):
  $env:PATH = "C:\\conda_envs\\opensg_2_0_env;...;" + $env:PATH   # see CLAUDE.md
  & "C:\\conda_envs\\opensg_2_0_env\\python.exe" kl_timo_paper\\jax\\run_iso_tube.py
"""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
LIB = os.path.join(CC, "examples", "TW-paper", "lib")
sys.path.insert(0, LIB)

from gen_meshes import gen_tube_yaml  # noqa: E402
from tube_lib import homog            # noqa: E402

# ---- report Case-1 geometry / material ----
R_MEAN = 5.0           # mid-wall (mean) radius [m]
H = 0.2                # wall thickness [m]
E = 3.44e9             # Pa
NU = 0.3
G = E / (2.0 * (1.0 + NU))
ISO = {"E": [E, E, E], "G": [G, G, G], "nu": [NU, NU, NU]}
LAYUP = [(0.0, H)]     # single isotropic ply (angle irrelevant)

# center reference: nodes on the mid-wall circle, ABD reference shifted OML->center
R_REF = R_MEAN
D_SHIFT = H / 2.0      # OML->center shift (single iso ply => B=0 at center)

LBL = ["C11(EA)", "C22(GA2)", "C33(GA3)", "C44(GJ)", "C55(EI2)", "C66(EI3)"]
# report Table 3.1 references (x10^6)
ANALYTIC = {"C11": 21606e6, "C22": 4157e6, "C33": 4157e6,
            "C44": 207650e6, "C55": 269680e6, "C66": 269680e6}
VABS = {"C11": 21622e6, "C22": 4205e6, "C33": 4205e6,
        "C44": 207299e6, "C55": 269935e6, "C66": 269935e6}
FENICS = {"C11": 21606e6, "C22": 4153e6, "C33": 4153e6,
          "C44": 207650e6, "C55": 269680e6, "C66": 269680e6}


def run(n_elem):
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "inputs")
    os.makedirs(datadir, exist_ok=True)
    yaml_path = os.path.join(datadir, "iso_tube_R5_h02_n%d.yaml" % n_elem)
    gen_tube_yaml(yaml_path, R_REF, layup=LAYUP, mat=ISO, n=n_elem, ccw=True)
    _RM, KF = homog(yaml_path, R_ref=R_REF, d_shift=D_SHIFT,
                    k22_mode="exact", e3="inward")
    return np.asarray(KF), yaml_path


def report(KF, n_elem):
    d = np.diag(KF)
    keys = ["C11", "C22", "C33", "C44", "C55", "C66"]
    print(f"\n=== JAX gradient-Kirchhoff (OpenSG-2.0), N={n_elem} elements ===")
    print(f"  full 6x6 order {['EA','GA2','GA3','GJ','EI2','EI3']}:")
    for i in range(6):
        print("   " + "".join(f"{KF[i, j]:13.4e}" for j in range(6)))
    print(f"\n  {'term':9s} {'JAX-KL':>12s} {'Analytic':>12s} "
          f"{'%err(an)':>9s} {'VABS':>12s} {'%err(VABS)':>11s}")
    for i, k in enumerate(keys):
        jx = d[i]
        an = ANALYTIC[k]; vb = VABS[k]
        ea = 100.0 * (jx - an) / an
        ev = 100.0 * (jx - vb) / vb
        print(f"  {LBL[i]:9s} {jx:12.4e} {an:12.4e} {ea:+8.2f}% "
              f"{vb:12.4e} {ev:+10.2f}%")
    print("\n  (values x10^6 for the paper table:)")
    for i, k in enumerate(keys):
        print(f"   {k:4s} JAX-KL = {d[i]/1e6:10.1f}   "
              f"FEniCS-KL = {FENICS[k]/1e6:10.1f}   "
              f"Analytic = {ANALYTIC[k]/1e6:10.1f}   "
              f"VABS = {VABS[k]/1e6:10.1f}")


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [64]
    for n in ns:
        KF, yp = run(n)
        report(KF, n)
        print(f"\n  yaml: {yp}")
