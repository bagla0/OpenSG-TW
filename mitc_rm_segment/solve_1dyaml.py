"""solve_1dyaml.py -- order/orientation-ROBUST multi-material RM Timo solve from a
1D boundary cross-section YAML (ours or OpenSG ShellSegmentMesh._create_1Dyaml).

Writer-independent canonicalization -- any two files describing the same boundary
give the SAME 6x6 to solver precision (~1e-7 of the matrix scale; verified on
BAR-URC seg5 L/R: JAX-extracted vs official OpenSG 1D YAML, max rel diff 4e-7):
  - edge DIRECTION: CCW about the section centroid (coordinates only -- the file's
    orientation rows are writer-convention-dependent, e.g. OpenSG keeps the mesh
    frame with the axial component FIRST while ours keeps it LAST);
  - per-edge hoop curvature k22: vertex turning angles averaged over BOTH edge
    ends (symmetric discrete curvature, direction-covariant: flip -> -k22);
    junction ends (deg != 2) contribute zero.

Usage: solve_1dyaml.py A.yaml [B.yaml]  -> prints diag (+ pairwise max rel diff).
"""
import os, sys
import numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, shift_abd_reference
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness
from opensg_jax.fe_jax.msg_rm_timo import timoshenko_rm


def _row(r):
    if isinstance(r, str):
        return r.split()
    if len(r) == 1 and isinstance(r[0], str):
        return r[0].split()
    return list(r)


def load_1d(fn):
    d = yaml.safe_load(open(fn))
    nodes = np.array([[float(v) for v in _row(r)] for r in d["nodes"]], float)
    elems = np.array([[int(float(v)) for v in _row(e)] for e in d["elements"]], int)
    if elems.min() >= 1:
        elems = elems - 1
    lab = np.zeros(len(elems), int)
    name2idx = {s["elementSet"]: i for i, s in enumerate(d["sections"])}
    labs_min = min(min(es["labels"]) for es in d["sets"]["element"] if es["labels"])
    off = 0 if labs_min == 0 else 1
    for es in d["sets"]["element"]:
        si = name2idx[es["name"]]
        for l in es["labels"]:
            lab[l - off] = si
    ori = np.array([[float(v) for v in _row(o)] for o in d["elementOrientations"]], float)
    return d, nodes, elems, lab, ori


def solve(fn, shear="mitc_both", center_ref=True, k22_mode="geom", flat_tol=1e-3,
          verbose=False):
    """CURVATURE argument k22_mode: 'geom' (default; symmetric discrete curvature,
    flat edges |k22|<flat_tol snapped to EXACTLY 0), 'uniform' (-1/R), 'zero'."""
    d, nodes, elems, lab, ori = load_1d(fn)
    nd2 = nodes[:, :2]                                   # [cross1, cross2, axial] convention
    # --- canonical edge orientation, WRITER-INDEPENDENT: CCW about the section
    # centroid (cross((mid - c), t) > 0).  Depends only on node coordinates, which
    # agree between any two writers of the same boundary, so every file yields the
    # IDENTICAL directed mesh -> identical assembly.  (Orientation rows in the file
    # are writer-convention-dependent -- OpenSG _create_1Dyaml keeps the mesh frame,
    # axial first -- so they are NOT used for direction.)
    c0 = nd2.mean(0)
    fl = 0
    for i, (a, b) in enumerate(elems):
        tv = nd2[b] - nd2[a]; rm = 0.5 * (nd2[a] + nd2[b]) - c0
        if float(rm[0] * tv[1] - rm[1] * tv[0]) < 0:
            elems[i] = [b, a]; fl += 1
    # --- material per section ---
    matmap = {m["name"]: {"E": m["elastic"]["E"], "G": m["elastic"]["G"], "nu": m["elastic"]["nu"]}
              for m in d["materials"]}
    D_by, G_by = {}, {}
    for si, sec in enumerate(d["sections"]):
        lay = sec["layup"]
        mn = [p[0] for p in lay]; th = [float(p[1]) for p in lay]; an = [float(p[2]) for p in lay]
        abd = np.asarray(compute_ABD_matrix(th, an, mn, matmap)[0])
        if center_ref:
            abd = shift_abd_reference(abd, 0.5 * sum(th))
        D_by[si] = abd
        G_by[si] = np.asarray(transverse_shear_stiffness(th, an, mn, matmap)[0])
    # --- per-edge k22 (the curvature argument) ---
    if k22_mode == "zero":
        k22 = np.zeros(len(elems))
    elif k22_mode == "uniform":
        c = nd2.mean(0); R = float(np.mean(np.hypot(nd2[:, 0] - c[0], nd2[:, 1] - c[1])))
        k22 = np.full(len(elems), -1.0 / R)
    else:
        # signed local curvature, CANONICAL + direction-COVARIANT: vertex turning
        # angles averaged over BOTH ends of the edge (standard discrete curvature).
        # Reversing an edge negates both end contributions exactly (k22 -> -k22, the
        # correct covariance for the local (t,n) frame), and the sample stays centred
        # on the edge -- so any stored node/edge ordering yields the same assembly.
        # Junction ends (deg != 2) contribute zero identically in every file.
        from collections import defaultdict
        inc = defaultdict(list)
        for i, (a, b) in enumerate(elems):
            inc[int(a)].append(i); inc[int(b)].append(i)

        def _turn(tin, tout):
            return np.arctan2(tin[0] * tout[1] - tin[1] * tout[0], np.dot(tin, tout))

        k22 = np.zeros(len(elems))
        for i, (a, b) in enumerate(elems):
            a = int(a); b = int(b)
            t1 = nd2[b] - nd2[a]; L1 = np.linalg.norm(t1); t1 = t1 / L1
            kap = []
            js = [j for j in inc[b] if j != i]           # end b: turning i -> next
            if len(js) == 1:
                j = js[0]
                c = int(elems[j][1]) if int(elems[j][0]) == b else int(elems[j][0])
                t2v = nd2[c] - nd2[b]; L2 = np.linalg.norm(t2v)
                kap.append(_turn(t1, t2v / L2) / (0.5 * (L1 + L2)))
            js = [j for j in inc[a] if j != i]           # end a: turning prev -> i
            if len(js) == 1:
                j = js[0]
                c = int(elems[j][1]) if int(elems[j][0]) == a else int(elems[j][0])
                t0v = nd2[a] - nd2[c]; L0 = np.linalg.norm(t0v)
                kap.append(_turn(t0v / L0, t1) / (0.5 * (L0 + L1)))
            if kap:
                k22[i] = -float(np.mean(kap))            # code's -1/R convention (CCW ring -> -1/R)
        k22[np.abs(k22) < flat_tol] = 0.0                # FLAT edges (webs/panels): exactly zero
    C6, Deff, V0, V1 = timoshenko_rm(nd2, elems, lab, D_by, G_by, k22,
                                     p=1, return_warp=True, shear=shear)
    C6 = np.asarray(C6)
    if verbose:
        print("  %s: %d nodes, %d edges (%d flipped), %d sections, k22[%s] range [%.3f, %.3f]"
              % (os.path.basename(fn), len(nd2), len(elems), fl, len(D_by), k22_mode, k22.min(), k22.max()))
    return C6


if __name__ == "__main__":
    LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
    files = sys.argv[1:]
    res = {}
    for fn in files:
        C6 = solve(fn, verbose=True)
        res[fn] = C6
        print("    diag x1e9 =", np.array2string(np.diag(C6) / 1e9, precision=4))
    if len(files) == 2:
        A, B = res[files[0]], res[files[1]]
        scale = np.max(np.abs(A))
        print("\nmax |A-B| / max|A| = %.3e   (target <= 1e-5)" % (np.max(np.abs(A - B)) / scale))
        print("per-diag rel diff:", np.array2string(np.abs(np.diag(A) - np.diag(B)) / np.abs(np.diag(A)), precision=2))
