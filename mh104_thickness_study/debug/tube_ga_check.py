"""Isolate the JAX Kirchhoff transverse-shear (V1) over-prediction on the iso TUBE, where the
analytic thin-tube answer is known: EA=E*A, GJ=G*2 pi R^3 t, EI=E*pi R^3 t, GA=k*G*A with k=0.5.
Also verify the _curvature_from_corners fix leaves the (ordered-loop) tube k22 = -1/R unchanged."""
import os, sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("opensg_jax", "aniso_tube"):
    sys.path.insert(0, os.path.join(CC, p))
import jax
jax.config.update("jax_enable_x64", True)
import yaml
from fe_jax.msg_hermite import solve_tw_from_yaml
from fe_jax.msg_mesh import load_yaml, read_mesh, mesh_curvature
from rm_tube import rm_tube_6x6

YAML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CC, "benchmark_tube", "data", "shell_iso_0.06.yaml")
FRAC = 0.5
print("tube YAML:", YAML)

d = yaml.safe_load(open(YAML))
nodes = np.array([[float(v) for v in (str(r[0]).split() if len(r) == 1 else r)] for r in d["nodes"]])
R = float(np.hypot(nodes[:, 0], nodes[:, 1]).mean())
t = float(sum(p[1] for p in d["sections"][0]["layup"]))
m = d["materials"][0]["elastic"]
E = float(m["E"][0]); G = float(m["G"][0]); nu = float(m["nu"][0])
A = 2 * np.pi * R * t
print("R=%.4f t=%.4f h/R=%.3f  E=%.3e G=%.3e nu=%.2f  A=%.4e" % (R, t, t / R, E, G, nu, A))

# --- curvature fix check: tube k22 must be -1/R (uniform) ---
n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
nd, cells, lpe = read_mesh(n3d, elements, e2l)
k22 = np.asarray(mesh_curvature(nd, cells, elements, is_closed=True))
print("k22: mean=%.4f  std=%.4f   (expect -1/R = %.4f)" % (k22.mean(), k22.std(), -1.0 / R))

# --- analytic thin-tube ---
EA = E * A; GJ = G * 2 * np.pi * R**3 * t; EI = E * np.pi * R**3 * t; GA = 0.5 * G * A
print("\nanalytic (k=0.5):  EA=%.4e  GA=%.4e  GJ=%.4e  EI=%.4e" % (EA, GA, GJ, EI))

Ck = np.asarray(solve_tw_from_yaml(YAML, frac=FRAC)["Timo"]); Ck = 0.5 * (Ck + Ck.T)
_rmout = rm_tube_6x6(YAML, FRAC, is_closed=True)
Cr = np.asarray(_rmout[0] if isinstance(_rmout, tuple) else _rmout); Cr = 0.5 * (Cr + Cr.T)


def line(tag, C):
    print("%-10s  EA=%.4e  GA2=%.4e  GA3=%.4e  GJ=%.4e  EI2=%.4e  EI3=%.4e" % (
        tag, C[0, 0], C[1, 1], C[2, 2], C[3, 3], C[4, 4], C[5, 5]))
    print("%-10s  GA2 vs analytic = %+.1f%% ,  GA3 = %+.1f%%" % (
        "", 100 * (C[1, 1] - GA) / GA, 100 * (C[2, 2] - GA) / GA))


print()
line("Kirchhoff", Ck)
line("RM", Cr)
