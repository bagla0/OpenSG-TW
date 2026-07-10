"""2_hybrid_tube_4way.py -- HYBRID hex+tet taper test + solid-vs-shell 4-way (tube m45).

One tapered-tube case (thick, m45, aR=0.7), four solid variants through the SAME JAX
solid-taper solver, then the RM SHELL ring + shell taper segment on the equivalent shell
mesh for comparison:

  solid HEX     : the structured hex mesh as generated (quad4 boundaries);
  solid HYBRID  : hexes with centroid x2>0 split into 6 tets each -> ONE segment with a
                  hex8 batch AND a tet4 batch, boundaries MIXED quad4+tri3;
  solid TET     : every hex split (tri3 boundaries);
  RM shell      : ring 6x6 (L, R) + shell taper segment 6x6 (mitc_rm_segment machinery).

Prints all boundary + segment 6x6 with wall times, then the comparison tables
(%err, VABS max/1000 neglect rule).  Writes tube_m45_4way.dat next to this script.

    python 2_hybrid_tube_4way.py [tag]           (default thick_m45_aR070)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "mitc_rm_segment"))

from opensg_jax.fe_jax.solid_taper import (read_solid_segment_yaml, split_batches_to_tets,
                                           compute_timo_taper_solid_seg)

ORDER = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
TAG = sys.argv[1] if len(sys.argv) > 1 else "thick_m45_aR070"
MESH = os.path.join(ROOT, "mitc_rm_segment", "out", "taper_study", "meshes")


def show(title, S, dt=None):
    print("\n%s%s" % (title, ("   [%.2f s]" % dt) if dt is not None else ""))
    for r in range(6):
        print("  " + "  ".join("% .5e" % S[r, c] for c in range(6)))
    print("  diagonal: " + "  ".join("%s=%.4e" % (ORDER[i], S[i, i]) for i in range(6)))


def pct(S, R):
    cut = np.abs(R).max() / 1e3                            # VABS max/1000 neglect rule
    E = np.full((6, 6), np.nan)
    m = np.abs(R) > cut
    E[m] = 100.0 * (S[m] - R[m]) / R[m]
    return E


def show_pct(title, E):
    print("\n%s  (%% err, VABS max/1000 rule; . = negligible term)" % title)
    for r in range(6):
        print("  " + "  ".join(("%8.3f" % E[r, c]) if np.isfinite(E[r, c]) else "     .  "
                               for c in range(6)))
    print("  max |err| = %.3f %%" % np.nanmax(np.abs(E)))


# ------------------------------------------------------------------ solid variants (JAX)
yaml_path = os.path.join(MESH, "solid_%s.yaml" % TAG)
seg_hex = read_solid_segment_yaml(yaml_path)
conn, _mat, _frm = seg_hex["batches"]["hex8"]
cent_x2 = seg_hex["nodes"][conn].mean(1)[:, 1]
seg_hyb = split_batches_to_tets(seg_hex, mask=cent_x2 > 0.0)   # half the hoop -> tets
seg_tet = split_batches_to_tets(seg_hex)                       # all tets

results = {}
for name, sg in (("HEX", seg_hex), ("HYBRID", seg_hyb), ("TET", seg_tet)):
    t0 = time.time()
    DL, DR, DS, info = compute_timo_taper_solid_seg(sg, verbose=False)
    dt = time.time() - t0
    bt = {k: len(v[0]) for k, v in sg["batches"].items()}
    print("\n================ solid %s : batches %s  dof=%d  %.2f s ================"
          % (name, bt, info["dof"], dt))
    show("%s L-boundary 6x6:" % name, DL, info["t_boundary"] / 2)
    show("%s R-boundary 6x6:" % name, DR, info["t_boundary"] / 2)
    show("%s SEGMENT 6x6:" % name, DS, info["t_segment"])
    results[name] = dict(L=DL, R=DR, seg=DS, dt=dt, info=info, batches=bt)

# ------------------------------------------------------------------ RM shell ring + segment
t0 = time.time()
from taper_study import shell_solve                        # mitc_rm_segment machinery
CL, S6, CR = shell_solve(TAG)
dt_shell = time.time() - t0
print("\n================ RM SHELL (mitc4_both) : %.2f s ================" % dt_shell)
show("SHELL ring L 6x6:", CL)
show("SHELL ring R 6x6:", CR)
show("SHELL taper segment 6x6:", S6)
results["SHELL"] = dict(L=CL, R=CR, seg=S6, dt=dt_shell)

# ------------------------------------------------------------------ comparison tables
print("\n############ COMPARISONS (tube %s) ############" % TAG)
show_pct("solid HYBRID vs solid HEX -- SEGMENT", pct(results["HYBRID"]["seg"], results["HEX"]["seg"]))
show_pct("solid TET    vs solid HEX -- SEGMENT", pct(results["TET"]["seg"], results["HEX"]["seg"]))
for part, nm in (("L", "L boundary/ring"), ("R", "R boundary/ring"), ("seg", "TAPER segment")):
    show_pct("RM SHELL vs solid HEX -- %s" % nm, pct(results["SHELL"][part], results["HEX"][part]))

dat = os.path.join(HERE, "tube_m45_4way.dat")
with open(dat, "w") as f:
    f.write("# tube %s : JAX solid taper (HEX | HYBRID hex+tet | TET) vs RM shell\n" % TAG)
    f.write("# rows: variant, then L / R / segment diagonals [EA GA2 GA3 GJ EI2 EI3] + wall time\n")
    for name in ("HEX", "HYBRID", "TET", "SHELL"):
        r = results[name]
        f.write("\n%s  (%.2f s)%s\n" % (name, r["dt"],
                ("  batches=%s" % r.get("batches")) if "batches" in r else ""))
        for part in ("L", "R", "seg"):
            f.write("  %-4s " % part + " ".join("%.6e" % r[part][i, i] for i in range(6)) + "\n")
    f.write("\n# %%err (max/1000 rule): HYBRID vs HEX seg max %.3f ; TET vs HEX seg max %.3f\n"
            % (np.nanmax(np.abs(pct(results["HYBRID"]["seg"], results["HEX"]["seg"]))),
               np.nanmax(np.abs(pct(results["TET"]["seg"], results["HEX"]["seg"])))))
print("\nwrote", dat)
