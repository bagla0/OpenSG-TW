"""compare_rm_3dfea.py -- OpenSG-RM shell vs the 3-D FEA benchmark.

    Abaqus_results/layup_db_abaqus.dat   (400 S4  + one homogenized 8x8)
    Abaqus_results/layup_db_3dfea.dat    (6400 C3D8I, ply by ply)
                     ->  layup_db_compare.out  +  layup_db_compare.png

Both decks come from the same layup_db.yaml and share geometry, load, BCs and
time integration, so the only difference is how the wall is represented.

THE SIGN: the shell is loaded with P on a +z-normal S4 and deflects +z; the
solid is loaded with P2 on its top face, which acts -z.  The solid history is
therefore negated here before anything is compared.  See
abaqus_3dfea_README.md.

THE STATION: the solid now reports NTOP3D, the centre of the TOP surface
(Nayak's station).  The shell's NCEN_REF is its REFERENCE surface.  Those are
not the same plane, and the difference is exactly the thickness stretch an RM
shell sets to zero -- so the shell has to be recovered to z = +h/2 from the
PATCHC resultants before the comparison means anything.  That recovery is NOT
wired in yet; this script warns and reports the mismatched pair meanwhile.

Run:  python compare_rm_3dfea.py
"""
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "Abaqus_results")
STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5, mfc="none",
               mew=1.2, markevery=(0, 32))
STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.4, ms=4, mfc="none",
              mew=1.1, markevery=(12, 32))


def history(dat, nset, comp=2):
    """Centre-node history: one value per increment, plus the step times."""
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


t_rm, w_rm = history(os.path.join(RES, "layup_db_abaqus.dat"), "NCEN_REF")
t_so, w_so = history(os.path.join(RES, "layup_db_3dfea.dat"), "NTOP3D")
w_so = -w_so                       # P2 acts -z; bring the solid into +z

# !! STATIONS DO NOT MATCH YET !!  NCEN_REF is the shell's REFERENCE surface,
# NTOP3D is the solid's TOP surface.  The difference between them contains the
# through-thickness compression, which an RM shell cannot represent (eps33 = 0).
# To compare like for like the shell must be dehomogenized to z = +h/2 from the
# PATCHC resultants; until that is wired in, read the numbers below as
# shell-reference vs solid-top, NOT as an OpenSG-RM error.
print("WARNING: comparing the shell's REFERENCE surface against the solid's "
      "TOP surface.\n         The gap includes thickness stretch. Recover the "
      "shell to z = +h/2 first.\n")

n = min(len(w_rm), len(w_so))
assert np.allclose(t_rm[:n], t_so[:n]), "the two decks did not share increments"
t, w_rm, w_so = t_rm[:n], w_rm[:n], w_so[:n]

pk_rm, pk_so = int(np.argmax(w_rm)), int(np.argmax(w_so))
err_pk = 100 * (w_rm[pk_rm] - w_so[pk_so]) / w_so[pk_so]
rms = 100 * np.sqrt(np.mean((w_rm - w_so) ** 2)) / w_so.max()

lines = [
    "OpenSG-RM shell vs 3-D FEA benchmark -- centre deflection",
    "  shell : 400 S4, one homogenized 8x8 plate law",
    "  solid : 6400 C3D8I, 9 materials ply by ply (history negated: P2 acts -z)",
    "",
    "%-28s %14s %14s %10s" % ("", "OpenSG-RM", "3-D FEA", "diff"),
    "%-28s %14.6f %14.6f %9.2f%%"
    % ("peak deflection [m]", w_rm[pk_rm], w_so[pk_so], err_pk),
    "%-28s %14.4f %14.4f" % ("  at t [ms]", 1e3 * t[pk_rm], 1e3 * t[pk_so]),
    "%-28s %14.6f %14.6f" % ("mean (static) level [m]", w_rm.mean(),
                             w_so.mean()),
    "",
    "%-28s %9.2f%%" % ("RMS difference / peak", rms),
    "%-28s %14d" % ("increments compared", n),
]
out = os.path.join(HERE, "layup_db_compare.out")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(1e3 * t, w_so, label="Abaqus 3-D FEA", **STY_SOL)
ax.plot(1e3 * t, w_rm, label="OpenSG-RM", **STY_RM)
ax.set_xlabel("time [ms]", fontsize=11)
ax.set_ylabel("centre deflection $U_3$ [m]", fontsize=11)
ax.set_xlim(0, 1e3 * t[-1])
ax.grid(alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "layup_db_compare.png"), dpi=150,
            bbox_inches="tight")
print("\nwrote layup_db_compare.out + .png")
