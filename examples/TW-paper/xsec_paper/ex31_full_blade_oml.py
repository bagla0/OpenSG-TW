"""ex31_full_blade_oml.py -- EXAMPLE 3.1: full IEA-22 blade, RM 6-DOF ring vs 2-D solid
at every span station, with the CORRECT reference convention for the stored shell yamls:
the contour is the OML (outer mold line), so the laminate is stacked INWARD from the
mesh line (center_ref=False), exactly like official OpenSG shell frac=0 and the st15 /
BAR-URC airfoil convention.  (center_ref=True centred the laminate on the OML line,
pushing half the wall outside the airfoil -> the spurious +8..16% GJ/EI2.)

Step 1 verifies the convention at r=0.2: center vs OML(raw) vs OML(shifted the other
way), against the PreVABS 2-D solid; picks the physically consistent one.
Step 2 runs all stations r=0.2..0.9 with it -> npz + spanwise figure + KL-paper-style
per-station data.

  -> results/ex31_full_blade.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for q in (MITC, REPO):
    sys.path.insert(0, q)

import yaml
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from opensg_jax.fe_jax.msg_materials import shift_abd_reference
from run_ring_indep import ring_indep
from xsec_5v6_master import _row, _norm_materials, load_solid, LBL

IB = os.path.abspath(os.path.join(HERE, "..", "iea22_blade", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
STATIONS = [("r020", 0.2), ("r030", 0.3), ("r040", 0.4), ("r050", 0.5),
            ("r060", 0.6), ("r070", 0.7), ("r080", 0.8), ("r090", 0.9)]


def load_ring_ref(path, ref="oml"):
    """ring arrays with an explicit laminate reference: 'center' | 'oml' (stack from the
    line, raw ABD) | 'oml_flip' (reference shifted the full thickness the other way)."""
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re3 = ori[:, 6:9]
    sections = d["sections"]; materials = _norm_materials(d["materials"])
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    if ref == "center":
        D_by, G_by = _material_by_section(sections, materials, center_ref=True)
    else:
        D_by, G_by = _material_by_section(sections, materials, center_ref=False)
        if ref == "oml_flip":
            for si, sec in enumerate(sections):
                t = sum(float(p[1]) for p in sec["layup"])
                D_by[si] = shift_abd_reference(np.asarray(D_by[si]), t)
    k22 = compute_k22(rx[cells].mean(1), ori[:, 3:6], re3, cells)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, D_by=D_by, G_by=G_by,
                k22=k22, ax=2, cross=[0, 1])


def c6(R):
    C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                   R["k22"], R["ax"], R["cross"], shear="mitc4_g23", lam_space="elem")
    return 0.5 * (C + C.T)


def derr(C, So):
    return np.array([100.0 * (C[i, i] - So[i, i]) / So[i, i] for i in range(6)])


# ---------------- step 1: convention check at r=0.2 ----------------
So2 = load_solid(os.path.join(IB, "C6_solid_r020.txt"))
sp2 = os.path.join(IB, "shell_r020.yaml")
print("=== r=0.2 laminate-reference check (diag %err vs PreVABS 2-D solid) ===")
best, bestref = None, None
for ref in ("center", "oml", "oml_flip"):
    C = c6(load_ring_ref(sp2, ref))
    e = derr(C, So2)
    print("  %-9s %s" % (ref, "  ".join("%s%+7.2f" % (LBL[i], e[i]) for i in range(6))), flush=True)
    score = np.mean(np.abs(e))
    if best is None or score < best:
        best, bestref = score, ref
print("--> convention with lowest mean |err|: %s (%.2f%%)" % (bestref, best))

# ---------------- step 2: full blade with the verified convention ----------------
rows, diag_err, tt = [], [], []
import time
for tag, r in STATIONS:
    shell = os.path.join(IB, "shell_%s.yaml" % tag)
    solid = os.path.join(IB, "C6_solid_%s.txt" % tag)
    if not (os.path.exists(shell) and os.path.exists(solid)):
        print("skip", tag); continue
    So = load_solid(solid)
    t0 = time.time(); C = c6(load_ring_ref(shell, bestref)); dt = time.time() - t0
    e = derr(C, So); diag_err.append(e); tt.append(dt)
    rows.append((tag, r, So, C))
    print("r=%.1f  RM6DOF(%s) diag %%err: %s   [%.2fs]"
          % (r, bestref, "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6)), dt), flush=True)

R = [r for _t, r, _s, _c in rows]
E = np.array(diag_err)
np.savez(os.path.join(OUT, "ex31_full_blade.npz"),
         r=np.array(R), diag_err=E, labels=LBL, ref=bestref,
         solids=np.array([s for _t, _r, s, _c in rows]),
         rings=np.array([c for _t, _r, _s, c in rows]), times=np.array(tt))

MARK = ["o", "s", "^", "D", "v", "P"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
groups = [([1, 2, 3], "transverse shear & torsion"), ([0, 4, 5], "extension & bending")]
for a, (idx, title) in zip(ax, groups):
    a.axhline(0, color="0.6", lw=0.8)
    for k in idx:
        a.plot(R, E[:, k], color=COL[k], marker=MARK[k], lw=1.6, ms=5, label="$%s$" %
               ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"][k])
    a.set_xlabel("span station  $r$"); a.set_title(title); a.grid(alpha=0.3)
    a.legend(fontsize=9, frameon=False)
ax[0].set_ylabel("diagonal % error vs 2-D solid")
fig.tight_layout(); p = os.path.join(FIG, "full_blade_rm_span.png")
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print("\nwrote ex31_full_blade.npz +", p)
