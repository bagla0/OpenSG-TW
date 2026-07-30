"""compare_analytic_caseA.py -- MSG-RM vs the EXACT 3-D solution for Garg caseA
([0/90/0] Pagano, S = 10), with the section resultants taken from the EXACT solution
itself: the "reaction forces from Pagano" route -- no plate solver (Abaqus) needed.

REFERENCE SURFACE (one convention everywhere, stated explicitly): the laminate
MID-SURFACE (center reference).  The plate SG was generated with fraction = 0.5, so
x3 = 0 is the mid-plane; the Abaqus strip put its shell reference surface at the
mid-surface; and exact_cyl.profile() reports z from the mid-surface.  OML reference is
NOT used anywhere in this chain.

STATIONS (global plate coordinate x1, span a = 1 m), the standard Pagano choices:
    x1 = a/2   sigma11 / sigma33 / w through-thickness   (sin family peak)
    x1 = 0     sigma13 through-thickness                 (cos family peak)

FF FROM THE EXACT SOLUTION: integrate the exact stress amplitudes through the
thickness -- N = int s dz, M = int s z dz, Q = int t dz.  The sin-family amplitudes
ARE the x = a/2 values; the cos-family amplitude IS the x = 0 value.

sigma33: the first-order strain-gradient recovery has no surface-pressure column
(V1L/V2L deliberately unimplemented), so sigma33 is recovered the plate-consistent
way, by through-thickness equilibrium integration of the recovered sigma13 amplitude:
    d(s33_hat)/dz = p * s13_hat(z)     (families s13 ~ cos, s33 ~ sin)

Writes examples/garg/caseA/analytic_compare/: the .SM/.EM/.U/.out files at both
stations (via example 7, material frame), the comparison .dat, and the PNG plots
(global frame): s13 at x = 0, s33 and s11 at x = a/2.
"""
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CC)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CC, "examples", "TW-paper", "rm_thickness"))

from exact_cyl import ExactCyl
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from garg_layups import MATERIAL_DB, LAYUPS, H

OUT = os.path.join(HERE, "caseA", "analytic_compare")
if not os.path.isdir(OUT):
    os.makedirs(OUT)
YAML = os.path.join(HERE, "caseA", "garg_A_sg.yaml")
PY = os.environ.get("PYTHON", sys.executable)

q0 = 1.0e4; a = 1.0; S = 10.0
p = np.pi / a
h = a / S
lay = LAYUPS["caseA"]
fr = [t / H for t in lay["thick"]]
thk = [f * h for f in fr]; ang = lay["angles"]; mats = lay["mat_names"]

# ------------------------------------------- exact solution + its section resultants
ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=q0)
zc, sig, eps, uvw = ex.profile(n_per_layer=81)          # AMPLITUDES, z from MID-surface
s11_a, s22_a, s33_a, s23_a, s13_a, s12_a = (sig[:, k] for k in range(6))

N11 = np.trapezoid(s11_a, zc); N22 = np.trapezoid(s22_a, zc)
M11 = np.trapezoid(s11_a * zc, zc); M22 = np.trapezoid(s22_a * zc, zc)
Q1 = np.trapezoid(s13_a, zc)
w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

# the MSG plate solution's own mid-span deflection (u2d for the .U recovery)
r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
D11 = float(r["ABDG"][3, 3]); G11 = float(r["ABDG"][6, 6])
w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)

FF_mid = [N11, N22, 0.0, M11, M22, 0.0, 0.0, 0.0]       # x = a/2 (sin peak; Q1(a/2)=0)
FF_end = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Q1, 0.0]        # x = 0   (cos peak; M(0)=0)
u2d_mid = [0.0, 0.0, w_msg]

rep = ["# MSG-RM vs EXACT 3-D (Pagano), Garg caseA [0/90/0], S = a/h = %g" % S,
       "# reference surface: MID-SURFACE (center) everywhere -- SG fraction=0.5,",
       "#   Abaqus shell reference = mid-surface, exact z from mid-surface.  No OML.",
       "# stations: x = a/2 (s11, s33, w) ; x = 0 (s13)",
       "# FF integrated from the EXACT stresses (the Pagano 'reaction forces'):",
       "#   FF_mid = [%s]" % ", ".join("%.6g" % v for v in FF_mid),
       "#   FF_end = [%s]" % ", ".join("%.6g" % v for v in FF_end),
       "#   u2d    = [0, 0, %.6e]   (the plate solution's own w at a/2)" % w_msg,
       "#   exact w(a/2, z=0) = %.6e ;  Abaqus shell U3 was 94.3695e-6" % w_ex,
       "#"]

# --------------------------- example-7 runs: the .SM/.EM/.U files at both stations
for base, FF, u2d in (("analytic_mid", FF_mid, u2d_mid),
                      ("analytic_end", FF_end, [0.0, 0.0, 0.0])):
    cmd = [PY, os.path.join(CC, "examples", "7_get_plateRM_dehom_using_1DSG.py"),
           "--yaml", YAML, "--FF"] + ["%.10g" % float(v) for v in FF] + \
          ["--u2d"] + ["%.10g" % float(v) for v in u2d] + ["--base", os.path.join(OUT, base)]
    subprocess.run(cmd, check=True, capture_output=True)
    rep.append("# wrote %s.{SM,EM,U,out}" % base)

# ------------------------------- global-frame MSG profiles for the plots and errors
S6 = np.linalg.inv(np.asarray(r["A6"]))
E6_mid = S6 @ np.array(FF_mid[:6])
dE1_end = S6 @ np.array([0, 0, 0, FF_end[6], 0, 0.0])
E6_end = S6 @ np.array(FF_end[:6])
z6 = np.zeros(6)

s11_m = np.empty_like(zc); s33eq = np.empty_like(zc); s13_m = np.empty_like(zc)
for i, z in enumerate(zc):
    _, Sig, _ = msgrm_strain_at_depth(r, z, E6_mid, z6, z6)          # x = a/2: no Q
    s11_m[i] = Sig[0]
    _, Sig, _ = msgrm_strain_at_depth(r, z, E6_end, dE1_end, z6)     # x = 0: Q-driven
    s13_m[i] = Sig[4]
# sigma33 amplitude by equilibrium integration of the recovered s13 amplitude
s33eq[0] = 0.0
s33eq[1:] = np.cumsum(0.5 * (p * s13_m[1:] + p * s13_m[:-1]) * np.diff(zc))

def relerr(m, e):
    return 100 * np.linalg.norm(m - e) / np.linalg.norm(e)

rep += ["#", "# rel L2 errors vs exact (global frame):",
        "#   s11 (x=a/2): %7.3f %%" % relerr(s11_m, s11_a),
        "#   s13 (x=0)  : %7.3f %%" % relerr(s13_m, s13_a),
        "#   s33 (x=a/2): %7.3f %%   (equilibrium-integrated; top-face value %.4f q0)"
        % (relerr(s33eq, s33_a), s33eq[-1] / q0),
        "#", "# columns: z[m]  s11_msg  s11_exact  s13_msg  s13_exact  s33_msg  s33_exact  [Pa]"]
np.savetxt(os.path.join(OUT, "compare_analytic.dat"),
           np.column_stack([zc, s11_m, s11_a, s13_m, s13_a, s33eq, s33_a]),
           header="\n".join(rep), fmt="%15.6e")

# ------------------------------------------------------------------------- plots
EX, MS = "k", "#ff7f0e"
for fname, msg, exa, lab, station in (
        ("s13_x0.png", s13_m, s13_a, r"$\sigma_{13}$  [Pa]", "x = 0"),
        ("s33_xa2.png", s33eq, s33_a, r"$\sigma_{33}$  [Pa]", "x = a/2"),
        ("s11_xa2.png", s11_m, s11_a, r"$\sigma_{11}$  [Pa]", "x = a/2")):
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    ax.plot(exa, zc / h, "-", color=EX, lw=2.0, label="exact 3-D (Pagano)")
    ax.plot(msg, zc / h, ":s", color=MS, ms=4, mfc="none", mew=1.2, lw=1.6,
            markevery=4, label="MSG-RM")
    ax.set_xlabel(lab + "   at " + station, fontsize=11)
    ax.set_ylabel("$z/h$", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

print("\n".join(rep))
print("wrote", os.path.relpath(OUT, CC))
