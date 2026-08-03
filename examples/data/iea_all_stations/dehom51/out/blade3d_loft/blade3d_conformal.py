"""blade3d_conformal.py -- ONE CONFORMAL lofted blade mesh: skin + shear webs share their junction nodes.

Approach (as requested): FIRST build a CONSISTENT 2-D cross-section TOPOLOGY that is identical at every
station -- the OML contour graph (N nodes, closed loop) PLUS each shear web as a chain of nodes whose two
END nodes ARE OML nodes (so the web meets the skin through SHARED nodes, exactly like the 2-D section).
THEN loft: because every station has the same node indexing, connecting station k to k+1 (with MPER
interpolated planes) and turning every section edge into a quad gives a single WATERTIGHT, CONFORMAL mesh
-- shared nodes at the web/skin junctions AND across every station interface, closed contour (no seam).

Nodes are coloured by the RM two-step dehom fields (material-frame stress + TOTAL local disp).
Output: vtk/blade3d_conformal.vtk  (pyvista PolyData quads; RM_{S11,S22,S12,u1,u2,u3} + RM_disp[vec]).
"""
import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
XSEC = os.path.join(REPO, "examples", "TW-paper", "xsec_paper")
sys.path.insert(0, REPO); sys.path.insert(0, IO); sys.path.insert(0, XSEC)
import jax; jax.config.update("jax_enable_x64", True)
import yaml
from scipy.spatial import cKDTree
import dehom_rm
from opensg_io import load_blade, build_cross_section
import pyvista as pv

WINDIO = os.path.join(ROOT, "IEA-22-280-RWT.yaml")
SHELLD = os.path.join(ROOT, "shell51", "1d_yaml")
FF_ALL = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))
BD_OUT = os.path.join(IEA, "beamdyn", "iea51rmc_bd_driver.out")
VTK = os.path.join(HERE, "vtk"); os.makedirs(VTK, exist_ok=True)
N, NW, MPER = 240, 12, 20                       # OML nodes, nodes-per-web (incl 2 shared ends), span/interval
NWI = NW - 2                                     # interior web nodes (the shared ends are OML nodes)
BLADE_LEN = 138.204
SVOIGT = {"S11": 0, "S22": 1, "S12": 5}
FIELDS = ["RM_S11", "RM_S22", "RM_S12", "RM_u1", "RM_u2", "RM_u3"]


def resample(xy, n):
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    c = np.vstack([xy, xy[0]])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]; d /= d[-1]
    t = np.linspace(0, 1, n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


def beam_kinematics(path, node):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]]); return u_g, C
    raise ValueError("no BeamDyn header")


def _row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


def wall_thickness(shell, contour):
    """local total laminate thickness (m) at each contour point, from the shell-YAML layup."""
    d = yaml.safe_load(open(shell))
    nd = np.array([_row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(x) for x in _row(e)] for e in d["elements"]]); cells -= cells.min()
    sect = {s["elementSet"]: sum(float(p[1]) for p in s["layup"]) for s in d["sections"]}
    et = np.zeros(len(cells))
    for grp in d["sets"]["element"]:
        if grp["name"] in sect:
            for lab in grp["labels"]:
                et[int(lab) - 1] = sect[grp["name"]]
    m = nd[cells].mean(1)
    return et[cKDTree(m).query(contour)[1]]


blade = load_blade(WINDIO)

# ---- fixed cross-section TOPOLOGY (identical at every station -> loftable & conformal) ----
refwebs = blade.webs_at(0.5)                      # web attachment fractions from a mid station
NWEB = len(refwebs)
web_ia = [int(round(w["s"] * N)) % N for w in refwebs]   # OML index where web w meets the SUCTION skin
web_ib = [int(round(w["e"] * N)) % N for w in refwebs]   # OML index where web w meets the PRESSURE skin
Ntot = N + NWEB * NWI


def wnode(w, j):
    return N + w * NWI + j                        # global index of interior node j of web w


elems = [(i, (i + 1) % N) for i in range(N)]      # CLOSED OML loop (no seam)
for w in range(NWEB):
    chain = [web_ia[w]] + [wnode(w, j) for j in range(NWI)] + [web_ib[w]]   # ends are OML nodes -> SHARED
    for k in range(len(chain) - 1):
        elems.append((chain[k], chain[k + 1]))
elems = np.array(elems, dtype=np.int64)
print("topology: %d OML + %d webs x %d interior = %d nodes, %d section edges (webs share OML ends %s/%s)"
      % (N, NWEB, NWI, Ntot, len(elems), web_ia, web_ib))

# ---- per-station node coordinates + RM fields (same node indexing every station) ----
sections = []
for i in range(51):
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(shell):
        print("  s%02d missing" % i); continue
    r = i / 50.0
    try:
        B = dehom_rm.build_rm_bundle(shell); FF = FF_ALL[i, 1:]
        oml = resample(np.asarray(build_cross_section(blade, r=r)["nodes"], float), N)   # OML outer surface
        P = np.zeros((Ntot, 2)); P[:N] = oml
        for w in range(NWEB):
            Pa = oml[web_ia[w]]; Pb = oml[web_ib[w]]                 # web ends ARE OML nodes -> shared junction
            tl = np.linspace(0, 1, NW)[1:-1]
            P[wnode(w, 0):wnode(w, 0) + NWI] = Pa[None, :] + tl[:, None] * (Pb - Pa)[None, :]
        S = np.asarray(dehom_rm.stress_at_points(B, P, beam_force_vabs=FF, frame="material", n_per_layer=4)["stress"]) / 1e6
        Wg = np.asarray(dehom_rm.disp_at_points(B, P, beam_force_vabs=FF))
        u_g, C = beam_kinematics(BD_OUT, i + 1)
        r3 = np.column_stack([np.zeros(Ntot), P[:, 0], P[:, 1]])
        U = u_g + (C @ (Wg + r3).T).T - r3
        F = {}
        for c in ("S11", "S22", "S12"):
            F["RM_" + c] = S[:, SVOIGT[c]]
        for j, c in enumerate(("u1", "u2", "u3")):
            F["RM_" + c] = U[:, j]
        sections.append((r, P, F))
        print("  s%02d r=%.3f ok" % (i, r))
    except Exception as e:
        print("  s%02d FAIL: %s" % (i, str(e)[:70]))

# ---- loft: MPER interpolated planes between stations (same topology -> conformal) ----
planes = []
for k in range(len(sections) - 1):
    r0, P0, F0 = sections[k]; r1, P1, F1 = sections[k + 1]
    for m in range(MPER):
        t = m / MPER
        planes.append(((r0 + t * (r1 - r0)) * BLADE_LEN, (1 - t) * P0 + t * P1,
                       {kk: (1 - t) * F0[kk] + t * F1[kk] for kk in F0}))
rL, PL, FL = sections[-1]; planes.append((rL * BLADE_LEN, PL, FL))
NS = len(planes)

# ---- single conformal quad PolyData: (Ntot x NS) shared nodes, one quad per (edge, span interval) ----
pts = np.zeros((Ntot * NS, 3)); fld = {kk: np.zeros(Ntot * NS) for kk in FIELDS}
for kp, (X, P, F) in enumerate(planes):
    base = kp * Ntot
    pts[base:base + Ntot, 0] = X; pts[base:base + Ntot, 1] = P[:, 0]; pts[base:base + Ntot, 2] = P[:, 1]
    for kk in FIELDS:
        fld[kk][base:base + Ntot] = F[kk]
faces = np.empty((elems.shape[0] * (NS - 1), 5), dtype=np.int64)
q = 0
for kp in range(NS - 1):
    b0 = kp * Ntot; b1 = (kp + 1) * Ntot
    for a, bb in elems:
        faces[q] = (4, b0 + a, b0 + bb, b1 + bb, b1 + a); q += 1
mesh = pv.PolyData(pts, faces.ravel())
for kk in FIELDS:
    mesh.point_data[kk] = fld[kk]
mesh.point_data["RM_disp"] = np.column_stack([fld["RM_u1"], fld["RM_u2"], fld["RM_u3"]])
out = os.path.join(VTK, "blade3d_conformal.vtk")
mesh.save(out)
print("\nwrote %s  (%d pts, %d quads; SINGLE conformal mesh, webs share skin nodes, closed loop)"
      % (os.path.relpath(out, HERE), mesh.n_points, mesh.n_cells))
