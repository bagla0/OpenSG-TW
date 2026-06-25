"""The IML (frac=1.0) breaks for thick walls because offset_oml_to_iml gives each node the AVERAGE
thickness + AVERAGE inward over all touching elements -- at a web/skin junction the thick perpendicular
web folds the geometry.  Test candidate fixes (skin/contour drives the junction offset) on f=0.2 (thin)
and f=0.6 (thick) IML, vs solid.  Monkey-patch the offset so no core edit until a winner is chosen."""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
import fe_jax.msg_hermite as mh
import fe_jax.msg_mesh as mm
from fe_jax.msg_hermite import solve_tw_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CC, "mh104_thickness_study", "results")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
orig = mm.offset_oml_to_iml


def _accum(nodes, cells, layup_per_elem, layup_db, elem_e3):
    n = nodes.shape[0]; cen = nodes.mean(0)
    acc_in = np.zeros((n, 2)); per = [[] for _ in range(n)]
    for e in range(cells.shape[0]):
        c0, c1 = int(cells[e, 0]), int(cells[e, -1]); seg = nodes[c1] - nodes[c0]; L = np.hypot(*seg)
        if L < 1e-30:
            continue
        if elem_e3 is not None:
            inw = np.asarray(elem_e3[e], float)
        else:
            t = seg / L; inw = np.array([-t[1], t[0]]); mid = 0.5 * (nodes[c0] + nodes[c1])
            if (cen - mid) @ inw < 0:
                inw = -inw
        h = float(sum(layup_db[layup_per_elem[e]]["thick"]))
        for c in (c0, c1):
            acc_in[c] += inw; per[c].append((inw, h, L))
    in_nrm = acc_in / (np.linalg.norm(acc_in, axis=1, keepdims=True) + 1e-30)
    return n, in_nrm, per


def cand_aligned(nodes_2d, cells, layup_per_elem, layup_db, elem_e3=None, frac=1.0):
    """Thickness from the element whose inward is MOST ALIGNED with the consensus (the smooth skin),
    not the average -> a junction offsets by the skin thickness, not the thick web."""
    nodes = np.asarray(nodes_2d, float); n, inw, per = _accum(nodes, cells, layup_per_elem, layup_db, elem_e3)
    th = np.zeros(n)
    for c in range(n):
        if per[c]:
            th[c] = max(per[c], key=lambda x: x[0] @ inw[c])[1]
    return nodes + frac * th[:, None] * inw


def cand_min(nodes_2d, cells, layup_per_elem, layup_db, elem_e3=None, frac=1.0):
    """Min thickness at each node (the thin skin caps the offset)."""
    nodes = np.asarray(nodes_2d, float); n, inw, per = _accum(nodes, cells, layup_per_elem, layup_db, elem_e3)
    th = np.array([min((x[1] for x in per[c]), default=0.0) for c in range(n)])
    return nodes + frac * th[:, None] * inw


def cand_cap(nodes_2d, cells, layup_per_elem, layup_db, elem_e3=None, frac=1.0):
    """Baseline offset, then cap each node displacement at 0.45 x its shortest element (anti-fold)."""
    out = np.asarray(orig(nodes_2d, cells, layup_per_elem, layup_db, elem_e3, frac), float)
    nodes = np.asarray(nodes_2d, float); n, inw, per = _accum(nodes, cells, layup_per_elem, layup_db, elem_e3)
    disp = out - nodes
    for c in range(n):
        if not per[c]:
            continue
        cap = 0.45 * min(x[2] for x in per[c]); d = np.hypot(*disp[c])
        if d > cap > 0:
            disp[c] *= cap / d
    return nodes + disp


def cand_alcap(nodes_2d, cells, layup_per_elem, layup_db, elem_e3=None, frac=1.0):
    """aligned (skin-consensus thickness) THEN cap displacement at 0.45 x shortest element (anti-fold)."""
    out = np.asarray(cand_aligned(nodes_2d, cells, layup_per_elem, layup_db, elem_e3, frac), float)
    nodes = np.asarray(nodes_2d, float); n, inw, per = _accum(nodes, cells, layup_per_elem, layup_db, elem_e3)
    disp = out - nodes
    for c in range(n):
        if not per[c]:
            continue
        cap = 0.45 * min(x[2] for x in per[c]); d = np.hypot(*disp[c])
        if d > cap > 0:
            disp[c] *= cap / d
    return nodes + disp


def run(fn, fi):
    if fn is not None:
        mm.offset_oml_to_iml = fn; mh.offset_oml_to_iml = fn
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    C = np.asarray(solve_tw_from_yaml(shy, frac=1.0)["Timo"]); return 0.5 * (C + C.T)


for fi in (20, 60):
    S = np.loadtxt(os.path.join(RES, "C6_solid_f%03d.txt" % fi))
    print("\n=== f=0.%d  IML diagonal %%diff vs solid ===" % (fi // 10))
    print("  %-10s " % "cand" + " ".join("%-7s" % l for l in lab))
    for name, fn in [("baseline", orig), ("aligned", cand_aligned), ("min", cand_min), ("alcap", cand_alcap)]:
        C = run(fn, fi)
        print("  %-10s " % name + " ".join("%+6.1f" % (100 * (C[i, i] - S[i, i]) / abs(S[i, i])) for i in range(6)))
