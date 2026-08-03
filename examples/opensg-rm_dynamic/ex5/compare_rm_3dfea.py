"""compare_rm_3dfea.py -- OpenSG-RM vs the 3-D FEA benchmark, BOTH at the
centre of the TOP SURFACE (Nayak's station).

    Abaqus_results/layup_db_abaqus.dat   400 S4  + one homogenized 8x8
    Abaqus_results/layup_db_3dfea.dat    6400 C3D8I, ply by ply
                     ->  layup_db_compare.out  +  layup_db_compare.png

THE STATION.  The solid reads its node at z = +h/2 directly.  The shell cannot:
an RM shell has ONE w per point (eps33 = 0 by construction) and it belongs to
the REFERENCE surface.  So the RM side is DEHOMOGENIZED to z = +h/2:

    U3(top) = w_node  +  w3(+h/2)

where w3 is the third component of the SG warping, driven by the plate strains,
their in-plane gradients and the face-pressure ladder qt6.  That warping term IS
the through-thickness compression -- the physics a bare shell node cannot show.
Comparing the shell's node against the solid's top face instead would charge the
RM model for a term it never claimed to carry, and the gap would read as an
OpenSG-RM error when it is a statement about plate kinematics.

The strain GRADIENTS the recovery needs cannot come from one element, which is
why the deck prints SF/SM over a 2x2 patch: the 4 elements x 4 integration
points are least-squares fitted to give the value and both first derivatives at
the centre.

THE SIGN.  The shell is loaded with P on a +z-normal S4 and deflects +z; the
solid is loaded with P2 on its top face, which acts -z.  The solid history is
negated here.  See abaqus_3dfea_README.md.

Run:  python compare_rm_3dfea.py
"""
import os
import re
import sys

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "yu2003"))

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_warping_at_depth
from recover_6p2 import read_elprint_tables

RES = os.path.join(HERE, "Abaqus_results")
STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5, mfc="none",
               mew=1.2, markevery=(0, 32))
STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.4, ms=4, mfc="none",
              mew=1.1, markevery=(12, 32))

# ---- case data, from the one input file ------------------------------------
db = yaml.safe_load(open(os.path.join(HERE, "layup_db.yaml")))
p = db["plate"]
A, B, Q0 = float(p["a"]), float(p["b"]), float(p["q0"])
fraction = float(db["fraction"])
P = np.pi / A                       # single Navier mode: q ~ sin(Px) sin(Py)

inp = read_plate_sg_yaml(os.path.join(HERE, "1dsg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], n_per_layer=inp["n_per_layer"],
                 elem_order=inp["elem_order"], fraction=fraction)
S6 = np.linalg.inv(np.asarray(r["A6"]))
h = float(sum(inp["thick"]))
ZTOP = (1.0 - fraction) * h - 1e-9  # top face, in the SG's own z (0 = ref plane)


def node_history(dat, nset, comp=2):
    """One value per increment from a *NODE PRINT table, plus the step times."""
    lines = open(dat, errors="replace").read().splitlines()
    w, active, seen = [], False, False
    for ln in lines:
        if ("NODE SET %s" % nset) in ln and "TABLE IS PRINTED" in ln:
            active, seen = True, False
            continue
        tok = ln.split()
        if not tok:
            continue
        if active and not seen:
            if tok[0] == "NODE":
                seen = True
            continue
        if active and seen and re.fullmatch(r"\d+", tok[0]):
            v = []
            for t in tok[1:]:
                try:
                    v.append(float(t))
                except ValueError:
                    pass
            if len(v) > comp:
                w.append(v[comp])
            active = False
    t = np.array([float(ln.split()[-1]) for ln in lines
                  if "STEP TIME COMPLETED" in ln])
    w = np.array(w)
    return t[len(t) - len(w):], w


def patch_fit(dat, name, x0):
    """Least-squares fit of the 8 section resultants over the 2x2 patch:
    per increment (value, d/dx, d/dy) at the station x0 -> (n_inc, 8, 3)."""
    tables = read_elprint_tables(dat)
    sf = xy = None
    for (es, labels), rows in tables.items():
        if es != name:
            continue
        ofs = rows.shape[1] - len(labels)
        if "SF1" in labels:
            idx = [labels.index(k) + ofs for k in
                   ("SF1", "SF2", "SF3", "SF4", "SF5", "SM1", "SM2", "SM3")]
            sf = rows[:, idx].reshape(-1, 16, 8)
        if "COORD1" in labels:
            idx = [labels.index(k) + ofs for k in ("COORD1", "COORD2")]
            xy = rows[:16, idx]
    if sf is None or xy is None:
        raise RuntimeError("PATCHC: SF/SM or COORD table missing from %s" % dat)
    xi, eta = xy[:, 0] - x0[0], xy[:, 1] - x0[1]
    Bm = np.column_stack([np.ones(16), xi, eta, xi * eta, xi ** 2, eta ** 2])
    out = np.empty((sf.shape[0], 8, 3))
    for k in range(sf.shape[0]):
        C, *_ = np.linalg.lstsq(Bm, sf[k], rcond=None)
        out[k, :, 0], out[k, :, 1], out[k, :, 2] = C[0], C[1], C[2]
    return out


# ---- the two histories -----------------------------------------------------
SHELL = os.path.join(RES, "layup_db_abaqus.dat")
SOLID = os.path.join(RES, "layup_db_3dfea.dat")

t_rm, w_ref = node_history(SHELL, "NCEN_REF")     # reference-surface w
t_so, w_top_so = node_history(SOLID, "NTOP3D")
w_top_so = -w_top_so                              # P2 acts -z -> +z convention

fit = patch_fit(SHELL, "PATCHC", (A / 2, B / 2))
n = min(len(w_ref), len(w_top_so), fit.shape[0])
assert np.allclose(t_rm[:n], t_so[:n]), (
    "the decks did not share increments -- both need *DYNAMIC, DIRECT")

# ---- dehomogenize the shell to the top surface -----------------------------
R = [0, 1, 2, 5, 6, 7]              # N11,N22,N12,M11,M22,M12 of the 8
qt6 = Q0 * np.array([1.0, 0.0, 0.0, -P * P, 0.0, -P * P])   # ladder at centre
w_top_rm = np.empty(n)
for k in range(n):
    E6 = S6 @ fit[k][R, 0]
    dE1 = S6 @ fit[k][R, 1]
    dE2 = S6 @ fit[k][R, 2]
    d11 = d22 = -P * P * E6         # mode-consistent second gradients
    warp = msgrm_warping_at_depth(r, ZTOP, E6, dE1, dE2, d11,
                                  np.zeros(6), d22, qt6=qt6)
    w_top_rm[k] = w_ref[k] + float(np.asarray(warp)[2])

t, w_ref = t_rm[:n], w_ref[:n]
w_top_so = w_top_so[:n]

# ---- report ----------------------------------------------------------------
i_rm, i_so = int(np.argmax(w_top_rm)), int(np.argmax(w_top_so))
err = 100 * (w_top_rm[i_rm] - w_top_so[i_so]) / w_top_so[i_so]
rms = 100 * np.sqrt(np.mean((w_top_rm - w_top_so) ** 2)) / w_top_so.max()
stretch = w_top_rm - w_ref                      # what the recovery added

lines = [
    "OpenSG-RM vs 3-D FEA -- centre of the TOP surface (z = +h/2)",
    "  shell : 400 S4 + homogenized 8x8, DEHOMOGENIZED to z = +h/2",
    "  solid : 6400 C3D8I ply by ply, node at z = +h/2 (negated: P2 acts -z)",
    "",
    "%-30s %14s %14s %10s" % ("", "OpenSG-RM", "3-D FEA", "diff"),
    "%-30s %14.6f %14.6f %9.2f%%" % ("peak w at top [m]",
                                     w_top_rm[i_rm], w_top_so[i_so], err),
    "%-30s %14.4f %14.4f" % ("  at t [ms]", 1e3 * t[i_rm], 1e3 * t[i_so]),
    "%-30s %9.2f%%" % ("RMS difference / peak", rms),
    "",
    "the recovery's contribution (what a bare shell node cannot give):",
    "%-30s %14.6f" % ("  peak w at reference [m]", w_ref.max()),
    "%-30s %14.6f" % ("  peak warping w3(+h/2) [m]", stretch.max()),
    "%-30s %13.2f%%" % ("  i.e. of the top deflection",
                        100 * stretch.max() / w_top_rm[i_rm]),
    "",
    "%-30s %14d" % ("increments compared", n),
]
open(os.path.join(HERE, "layup_db_compare.out"), "w").write(
    "\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(1e3 * t, w_top_so, label="Abaqus 3-D FEA", **STY_SOL)
ax.plot(1e3 * t, w_top_rm, label="OpenSG-RM", **STY_RM)
ax.set_xlabel("time [ms]", fontsize=11)
ax.set_ylabel("top-surface deflection $U_3$ [m]", fontsize=11)
ax.set_xlim(0, 1e3 * t[-1])
ax.grid(alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "layup_db_compare.png"), dpi=150,
            bbox_inches="tight")
print("\nwrote layup_db_compare.out + .png")
