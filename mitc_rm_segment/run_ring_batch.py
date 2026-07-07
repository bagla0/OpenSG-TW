"""run_ring_batch.py <scratch_dir> -- RM boundary ring (6-DOF, FULL integration) for the
six PreVABS center-ref cases, compared to the 2-D solid boundary; flags shear locking
(thin transverse shears over-stiff)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import yaml
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from run_ring_indep import ring_indep

SP = sys.argv[1]
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def _norm_mats(mats):
    out = []
    for mm in mats:
        out.append(mm if "elastic" in mm else
                   {"name": mm["name"], "elastic": {"E": mm["E"], "G": mm["G"], "nu": mm["nu"]}})
    return out


def ring_full(shell_yaml, cx, cy):
    d = yaml.safe_load(open(shell_yaml))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    rx[:, 0] -= cx; rx[:, 1] -= cy
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re2, re3 = ori[:, 3:6], ori[:, 6:9]
    sections = d["sections"]; materials = _norm_mats(d["materials"])
    s2sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = s2sec[grp["name"]]
    D_by, G_by = _material_by_section(sections, materials, center_ref=True)
    k22 = compute_k22(rx[cells].mean(1), re2, re3, cells)
    C = ring_indep(rx, cells, rsub, re3, D_by, G_by, k22, 2, [0, 1], shear="full")
    return 0.5 * (C + C.T)


cases = [
    ("circle thin",  "sh_circ_thin.yaml",   "so_circ_thin.npz"),
    ("circle thick", "sh_circ_thick.yaml",  "so_circ_thick.npz"),
    ("square thin",  "sh_sq_thin.yaml",     "so_sq_thin.npz"),
    ("square thick", "sh_sq_thick.yaml",    "so_sq_thick.npz"),
    ("ellipse thin", "shell_ellt_thin_cr.yaml",  "so_ell_thin.npz"),
    ("ellipse thick","shell_ellT_thick_cr.yaml", "so_ell_thick.npz"),
]
print("%-13s %5s %8s %8s %8s %8s %8s %8s" % ("case", "", *LBL))
for name, shy, son in cases:
    b = np.load(os.path.join(SP, son)); So = b["S"]; cen = b["cen"]
    Sh = ring_full(os.path.join(SP, shy), cen[0], cen[1])
    err = [100 * (Sh[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    print("%-13s %5s " % (name, "%err") + " ".join("%+8.1f" % e for e in err))
    np.savez(os.path.join(SP, son.replace("so_", "sh_").replace(".npz", "_ring.npz")),
             Sh=Sh, So=So, cen=cen)
