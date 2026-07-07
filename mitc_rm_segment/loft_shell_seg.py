"""loft_shell_seg.py <contour.yaml> <tag> <mesh_dir> <A0 A1 B0 B1> [NL] -- loft a PreVABS
1-D shell contour into a tapered quad shell surface (geometric frame e1=span, e2=hoop,
e3=normal), write shell_<tag>.yaml, and solve the 6-DOF RM segment at FULL integration.
Prints the tapered 6x6.
"""
import os, sys
import numpy as np
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)

CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
SRC, TAG, MDIR = sys.argv[1], sys.argv[2], sys.argv[3]
A0, A1, B0, B1 = [float(v) for v in sys.argv[4:8]]
NL = int(sys.argv[8]) if len(sys.argv) > 8 else 12
# segment shear scheme: DEFAULT = canonical MITC (tie both gamma_13 and gamma_23);
# pass "full" as the 9th arg for full integration (no MITC).
SHEAR = sys.argv[9] if len(sys.argv) > 9 else "mitc4_both"
SOLVER = sys.argv[10] if len(sys.argv) > 10 else "auto"   # auto | dense | sparse
MERGE = (len(sys.argv) > 11 and sys.argv[11] == "merge")   # merge skin+web into 1 section
L = 2.0
os.makedirs(MDIR, exist_ok=True)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


d = yaml.load(open(SRC), Loader=CL)
cx0 = np.array([_row(r)[:2] for r in d["nodes"]])
cells = np.array([[int(v) - 1 for v in _row(e)] for e in d["elements"]])
cori = np.array([_row(o) for o in d["elementOrientations"]])
ce3 = cori[:, 6:9]
m = len(cx0); ne = len(cells)
cen = cx0.mean(0)
xc = cx0[:, 0] - cen[0]; yc = cx0[:, 1] - cen[1]

# per-element section id
sections = d["sections"]
s2sec = {s["elementSet"]: i for i, s in enumerate(sections)}
esec = np.zeros(ne, dtype=int)
for grp in d["sets"]["element"]:
    for lab in grp["labels"]:
        esec[int(lab) - 1] = s2sec[grp["name"]]

Z = np.linspace(0, L, NL + 1)
def sc(z):
    s = z / L
    return (A0 + (A1 - A0) * s) / A0, (B0 + (B1 - B0) * s) / B0

# nodes: NL+1 rings of the m contour nodes, tapered
snodes = []
for z in Z:
    fa, fb = sc(z)
    for i in range(m):
        snodes.append([float(xc[i] * fa), float(yc[i] * fb), float(z)])

squads = []; soris = []; slabels = {}
for i in range(NL):
    for ei, (a, b) in enumerate(cells):
        n_b0 = int(i * m + a); n_b1 = int(i * m + b)
        n_t1 = int((i + 1) * m + b); n_t0 = int((i + 1) * m + a)
        squads.append([n_b0 + 1, n_b1 + 1, n_t1 + 1, n_t0 + 1])
        P = np.array(snodes)
        e2 = _norm(P[n_b1] - P[n_b0])                       # hoop tangent
        gen = 0.5 * ((P[n_t0] - P[n_b0]) + (P[n_t1] - P[n_b1]))
        e1 = _norm(gen - (gen @ e2) * e2)                   # span (taper-tilted)
        e3 = _norm(np.cross(e1, e2))
        if esec[ei] > 0:                                    # web: e3 toward +x (gen_ell3w
            if e3[0] < 0:                                   # convention; a consistent web
                e3 = -e3; e1 = -e1                          # director across the T-junction)
        else:                                               # skin: e3 inward (contour e3)
            ref = np.array([ce3[ei, 0], ce3[ei, 1], 0.0])
            if e3 @ ref < 0:
                e3 = -e3; e1 = -e1
        soris.append([float(v) for v in np.concatenate([e1, e2, e3])])
        slabels.setdefault(esec[ei], []).append(len(squads))

# sections/materials in gen-style shell format (layup [mat,t,angle]; materials with elastic)
mats = []
for mm in d["materials"]:
    if "elastic" in mm:
        mats.append({"name": mm["name"], "density": mm.get("density", 1800.0), "elastic": mm["elastic"]})
    else:
        mats.append({"name": mm["name"], "density": mm.get("rho", 1800.0),
                     "elastic": {"E": mm["E"], "G": mm["G"], "nu": mm["nu"]}})
if MERGE:                                                   # all quads -> one section
    allq = list(range(1, len(squads) + 1))
    secs = [{"elementSet": "wall", "layup": sections[0]["layup"]}]
    setblk = [{"name": "wall", "labels": allq}]
else:
    secs = [{"elementSet": "layup_%d" % s2sec[s["elementSet"]], "layup": s["layup"]} for s in sections]
    setblk = [{"name": "layup_%d" % k, "labels": slabels.get(k, [])} for k in range(len(sections))]

shell = {"nodes": snodes, "elements": squads, "sections": secs,
         "sets": {"element": setblk}, "materials": mats, "elementOrientations": soris}
fn = os.path.join(MDIR, "shell_%s.yaml" % TAG)
yaml.safe_dump(shell, open(fn, "w"), default_flow_style=None, sort_keys=False)
print("wrote %s: %d nodes, %d quads, %d sections" % (fn, len(snodes), len(squads), len(sections)))

# force FULL integration and solve
import segment_indep
orig = segment_indep.assemble_segment_indep
def patched(*a, **k):
    k["shear"] = SHEAR; return orig(*a, **k)
segment_indep.assemble_segment_indep = patched
import importlib, run_indep as ri
importlib.reload(ri)
if SOLVER == "dense" or (SOLVER == "auto" and len(snodes) < 1500):
    solve = ri.shell_solve_lagrange
else:
    solve = getattr(ri, "shell_solve_lagrange_sparse", ri.shell_solve_lagrange)
print("solver:", solve.__name__)
Sh = np.asarray(solve(TAG, MDIR, os.path.join(MDIR, "res")))
segment_indep.assemble_segment_indep = orig
np.set_printoptions(suppress=True, linewidth=160)
print("\nRM SHELL SEGMENT (full) 6x6 (x1e9):\n", np.round(Sh / 1e9, 4))
print("diag:", "  ".join("%s=%.4g" % (LBL[i], Sh[i, i] / 1e9) for i in range(6)))
np.save(os.path.join(MDIR, "segsh_%s.npy" % TAG), Sh)
