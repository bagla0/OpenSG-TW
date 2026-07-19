"""blade_buckling.py -- IEA-22-280-RWT full-blade shell buckling under a flapwise surface traction,
root clamped / tip free.  Two pre-buckling stress pathways feed the SAME MITC4 shell eigensolver:

  JAX-FEA   : direct shell static solve K u = f (1500 Pa flapwise traction) -> element membrane N.
  RM-OpenSG : per-station RM homogenization; the cantilever beam internal force (the reduced resultant of
              the SAME traction) drives the two-step dehomogenization -> wall membrane N.

Mesh: ONE conformal skin+web shell-quad loft (blade3d_conformal topology: N OML nodes + 3 webs whose ends
ARE OML nodes -> shared junctions; watertight, closed loop).  ABD per element by the per-station layup
lookup (NOT a full-mesh reprojection): each section element -> layup -> mid-ref ABD (emitted yaml),
linearly interpolated along the span between real stations (a vanishing web would taper to a floor).

Usage:  python blade_buckling.py [mesh|fea|rm|all]   (default all).  Caches homogenization bundles."""
import os, sys, time, json, pickle
import numpy as np
from scipy.spatial import cKDTree
BUCK = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BUCK, "..", ".."))
IO = os.path.join(ROOT, "third_party", "OpenSG_io")
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")
IEA = os.path.join(ROOT, "examples", "data", "iea_all_stations")
sys.path.insert(0, ROOT); sys.path.insert(0, IO); sys.path.insert(0, XSEC); sys.path.insert(0, BUCK)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import yaml
import shell_buckling as sb
import dehom_rm
from emit_abd import load_station_abd
from opensg_io import load_blade, build_cross_section

WINDIO = os.path.join(IEA, "IEA-22-280-RWT.yaml")
SHELLD = os.path.join(IEA, "shell51", "1d_yaml")
ABDD = os.path.join(IEA, "dehom51", "out", "abd")            # iea_sNN_abd.yaml (51)
DATA = os.path.join(BUCK, "data"); CACHE = os.path.join(DATA, "blade_cache"); os.makedirs(CACHE, exist_ok=True)
BLADE_LEN = 138.204
N, NW, MPER = 120, 6, 2                                       # OML nodes, nodes/web, planes per station interval
NWI = NW - 2
PRESS = 1500.0                                                # Pa flapwise (global y) surface traction
NMODES = 8
NSTA = 51

blade = load_blade(WINDIO)


def resample(xy, n):
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    c = np.vstack([xy, xy[0]])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]; d /= d[-1]
    t = np.linspace(0, 1, n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


def _row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


# ---- fixed section topology (mid-span web attachment, 3 webs persist all stations) ----
refwebs = blade.webs_at(0.5); NWEB = len(refwebs)
web_ia = [int(round(w["s"] * N)) % N for w in refwebs]
web_ib = [int(round(w["e"] * N)) % N for w in refwebs]
Ntot = N + NWEB * NWI


def wnode(w, j):
    return N + w * NWI + j


sec_elems = [(i, (i + 1) % N) for i in range(N)]             # closed OML loop
is_web = [False] * N
for w in range(NWEB):
    chain = [web_ia[w]] + [wnode(w, j) for j in range(NWI)] + [web_ib[w]]
    for k in range(len(chain) - 1):
        sec_elems.append((chain[k], chain[k + 1])); is_web.append(True)
sec_elems = np.array(sec_elems); is_web = np.array(is_web); NSE = len(sec_elems)


def station_layup_lookup(shell_yaml):
    """KDTree of the station's native element midpoints -> layup name (covers skin AND web elements)."""
    d = yaml.safe_load(open(shell_yaml))
    nd = np.array([_row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(x) for x in _row(e)] for e in d["elements"]]); cells -= cells.min()
    name_of = {}
    for grp in d["sets"]["element"]:
        for lab in grp["labels"]:
            name_of[int(lab) - 1] = grp["name"]
    mids = nd[cells].mean(1)
    names = [name_of.get(i, d["sections"][0]["elementSet"]) for i in range(len(cells))]
    tree = cKDTree(mids)
    return tree, names


def station_abd(i):
    """P (Ntot,2) section coords, ABD (NSE,6,6), Gs (NSE,2,2) for real station i, via per-station layup lookup."""
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    r = i / 50.0
    oml = resample(np.asarray(build_cross_section(blade, r=r)["nodes"], float), N)
    P = np.zeros((Ntot, 2)); P[:N] = oml
    for w in range(NWEB):
        Pa, Pb = oml[web_ia[w]], oml[web_ib[w]]
        tl = np.linspace(0, 1, NW)[1:-1]
        P[wnode(w, 0):wnode(w, 0) + NWI] = Pa[None, :] + tl[:, None] * (Pb - Pa)[None, :]
    tree, names = station_layup_lookup(shell)
    ay = load_station_abd(os.path.join(ABDD, "iea_s%02d_abd.yaml" % i))["by_name"]
    ABD = np.zeros((NSE, 6, 6)); Gs = np.zeros((NSE, 2, 2))
    mids = 0.5 * (P[sec_elems[:, 0]] + P[sec_elems[:, 1]])
    idx = tree.query(mids + (tree.data.mean(0) - mids.mean(0)))[1]   # align conformal frame -> yaml frame
    for se in range(NSE):
        nm = names[idx[se]]
        A, G, thk = ay[nm]
        ABD[se] = A; Gs[se] = G
    return P, ABD, Gs


def build_blade(verbose=True):
    """Loft: per-station coords + ABD, MPER interpolated planes -> nodes, quads, per-elem ABD/Gs, root DOF."""
    t0 = time.time()
    Pk = np.zeros((NSTA, Ntot, 2)); Ak = np.zeros((NSTA, NSE, 6, 6)); Gk = np.zeros((NSTA, NSE, 2, 2))
    Rk = np.zeros(NSTA)
    for i in range(NSTA):
        P, A, G = station_abd(i)
        Pk[i] = P; Ak[i] = A; Gk[i] = G; Rk[i] = i / 50.0 * BLADE_LEN
        if verbose and i % 10 == 0:
            print("  station %d/%d built" % (i, NSTA))
    NS = (NSTA - 1) * MPER + 1
    pts = np.zeros((NS * Ntot, 3)); ABD_pl = np.zeros((NS, NSE, 6, 6)); Gs_pl = np.zeros((NS, NSE, 2, 2))
    Xpl = np.zeros(NS)
    for p in range(NS):
        rp = p / MPER; kL = min(int(np.floor(rp)), NSTA - 2); tt = rp - kL
        P = (1 - tt) * Pk[kL] + tt * Pk[kL + 1]
        X = (1 - tt) * Rk[kL] + tt * Rk[kL + 1]
        pts[p * Ntot:(p + 1) * Ntot, 0] = X
        pts[p * Ntot:(p + 1) * Ntot, 1] = P[:, 0]; pts[p * Ntot:(p + 1) * Ntot, 2] = P[:, 1]
        ABD_pl[p] = (1 - tt) * Ak[kL] + tt * Ak[kL + 1]; Gs_pl[p] = (1 - tt) * Gk[kL] + tt * Gk[kL + 1]
        Xpl[p] = X
    quads = []; ABD_e = []; Gs_e = []
    for p in range(NS - 1):
        b0, b1 = p * Ntot, (p + 1) * Ntot
        for se in range(NSE):
            a, bb = sec_elems[se]
            quads.append([b0 + bb, b1 + bb, b1 + a, b0 + a])     # cyclic rot of CCW quad: edge0->1 = SPAN (e1||laminate-1 fibers)
            ABD_e.append(0.5 * (ABD_pl[p, se] + ABD_pl[p + 1, se]))
            Gs_e.append(0.5 * (Gs_pl[p, se] + Gs_pl[p + 1, se]))
    quads = np.array(quads); ABD_e = np.array(ABD_e); Gs_e = np.array(Gs_e)
    root = np.unique([6 * n + k for n in range(Ntot) for k in range(6)])   # clamp root ring (plane 0)
    if verbose:
        print("blade mesh: %d nodes, %d quads, %d dof  (N=%d MPER=%d, %d planes)  built in %.1fs"
              % (len(pts), len(quads), 6 * len(pts), N, MPER, NS, time.time() - t0))
    return dict(nodes=pts, quads=quads, ABD_e=ABD_e, Gs_e=Gs_e, root=root, Xpl=Xpl, NS=NS,
                sec_mid_is_web=is_web, Pk=Pk, Ak=Ak, Rk=Rk)


def traction_load(nodes, quads):
    """1500 Pa flapwise (global +z) traction on every element -> consistent nodal load vector f (ndof,).
    Loft frame: x=span, y=chord (edgewise), z=airfoil-thickness (flapwise)."""
    ndof = 6 * len(nodes); f = np.zeros(ndof)
    fdir = np.array([0.0, 0.0, 1.0])
    for q in quads:
        X = nodes[q]
        area = 0.5 * np.linalg.norm(np.cross(X[2] - X[0], X[3] - X[1]))    # quad area
        fn = PRESS * area / 4.0
        for n in q:
            f[6 * n:6 * n + 3] += fn * fdir
    return f


def beam_forces_from_traction(nodes, f, Rk):
    """Cantilever internal force at each real station i: resultant of the traction outboard of X_i,
    reduced about the section reference (X_i,0,0).  Returned as [F1,F2,F3,M1,M2,M3] (span,chord,flap frame)."""
    fx = f.reshape(-1, 6)[:, :3]
    FF = np.zeros((NSTA, 6))
    for i in range(NSTA):
        Xi = Rk[i]; out = nodes[:, 0] >= Xi - 1e-6
        r = nodes[out] - np.array([Xi, 0.0, 0.0]); fn = fx[out]
        Fsum = fn.sum(0); Msum = np.cross(r, fn).sum(0)
        FF[i] = [Fsum[0], Fsum[1], Fsum[2], Msum[0], Msum[1], Msum[2]]
    return FF


def homogenize_station(i):
    """RM homogenization bundle for station i, cached to disk (build_rm_bundle is ~12s)."""
    cf = os.path.join(CACHE, "bundle_s%02d.pkl" % i)
    if os.path.exists(cf):
        return pickle.load(open(cf, "rb"))
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    B = dehom_rm.build_rm_bundle(shell)
    pickle.dump(B, open(cf, "wb"))
    return B


def rm_blade_N(bl, FF):
    """Per-station two-step dehom of the wall membrane N at each section element, then span loft to elements."""
    Pk, Ak = bl["Pk"], bl["Ak"]
    Nst = np.zeros((NSTA, NSE, 3)); t0 = time.time()
    for i in range(NSTA):
        B = homogenize_station(i)
        st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF[i])
        corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
        mids = 0.5 * (Pk[i][sec_elems[:, 0]] + Pk[i][sec_elems[:, 1]])
        shift = corners.mean(0) - Pk[i].mean(0)               # align conformal OML frame -> 1-D ring frame
        for se in range(NSE):
            e_ring, xi, pr = dehom_rm._project_point(corners, rc, mids[se] + shift)
            s6, _ = dehom_rm._rm_shell_strain(B, e_ring, xi, st_m, aA, aB)
            A = Ak[i][se][:3, :3]; Bm = Ak[i][se][:3, 3:]
            Nst[i][se] = A @ s6[:3] + Bm @ s6[3:6]            # [N11=span, N22=arc, N12]; e1=span so frame matches
        if i % 10 == 0:
            print("  dehom station %d/%d (%.0fs)" % (i, NSTA, time.time() - t0))
    # span loft: station N -> plane N -> element N (same interpolation as ABD)
    NS = bl["NS"]; Npl = np.zeros((NS, NSE, 3))
    for p in range(NS):
        rp = p / MPER; kL = min(int(np.floor(rp)), NSTA - 2); tt = rp - kL
        Npl[p] = (1 - tt) * Nst[kL] + tt * Nst[kL + 1]
    Ne = np.zeros((len(bl["quads"]), 3)); q = 0
    for p in range(NS - 1):
        for se in range(NSE):
            Ne[q] = 0.5 * (Npl[p, se] + Npl[p + 1, se]); q += 1
    return Ne, Nst, time.time() - t0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    bl = build_blade()
    nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
    np.savez(os.path.join(DATA, "blade_mesh.npz"), nodes=nodes, quads=quads, root=root,
             sec_is_web=bl["sec_mid_is_web"])
    if mode == "mesh":
        print("mesh only; saved blade_mesh.npz"); sys.exit(0)

    if mode in ("fea", "all"):
        t0 = time.time()
        K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
        f = traction_load(nodes, quads)
        u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
        Nvec = sb.element_membrane_N(nodes, quads, ABD_e, u)
        itip = int(np.argmax(nodes[:, 0]))
        print("FEA static: tip flap disp = %.4f m  (%.1fs); N11 range [%.3e, %.3e]"
              % (u[6 * itip + 2], time.time() - t0, Nvec[:, 0].min(), Nvec[:, 0].max()))
        t1 = time.time()
        loads_fea, modes_fea = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, root, n_modes=NMODES)
        print("FEA buckling: lambda_1 = %.4f  (%.1fs)  first: %s"
              % (loads_fea[0], time.time() - t1, np.array2string(loads_fea[:NMODES], precision=3)))
        np.savez(os.path.join(DATA, "blade_fea.npz"), loads=loads_fea, modes=modes_fea, Nvec=Nvec, u=u)

    if mode in ("rm", "all"):
        f = traction_load(nodes, quads)
        FF = beam_forces_from_traction(nodes, f, bl["Rk"])
        Ne, Nst, t_rm = rm_blade_N(bl, FF)
        print("RM dehom: N11 range [%.3e, %.3e]  (%.0fs homogenize+dehom)"
              % (Ne[:, 0].min(), Ne[:, 0].max(), t_rm))
        t1 = time.time()
        loads_rm, modes_rm = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Ne, root, n_modes=NMODES)
        print("RM buckling: lambda_1 = %.4f  (%.1fs)  first: %s"
              % (loads_rm[0], time.time() - t1, np.array2string(loads_rm[:NMODES], precision=3)))
        np.savez(os.path.join(DATA, "blade_rm.npz"), loads=loads_rm, modes=modes_rm, Nvec=Ne, FF=FF)

    if mode == "all":
        fea = np.load(os.path.join(DATA, "blade_fea.npz")); rm = np.load(os.path.join(DATA, "blade_rm.npz"))
        lf, lr = fea["loads"], rm["loads"]
        print("\n=== blade buckling: JAX-FEA vs RM-OpenSG ===")
        print("  lambda_1  FEA=%.4f  RM=%.4f  (RM/FEA=%.3f)" % (lf[0], lr[0], lr[0] / lf[0]))
        # N-field agreement (same quad frame now: [arc, span, shear])
        nf, nr = fea["Nvec"], rm["Nvec"]
        rel = np.linalg.norm(nr - nf) / (np.linalg.norm(nf) + 1e-30)
        cs = np.corrcoef(nf[:, 0], nr[:, 0])[0, 1]                # spanwise (axial) component (e1=span)
        print("  N field    ||RM-FEA||/||FEA|| = %.3f ; span-N corr = %.3f ; rms span FEA=%.3e RM=%.3e"
              % (rel, cs, np.sqrt((nf[:, 0]**2).mean()), np.sqrt((nr[:, 0]**2).mean())))
        json.dump({"lambda_FEA": [float(x) for x in lf], "lambda_RM": [float(x) for x in lr],
                   "N11_rel": float(rel), "mesh": [int(N), int(MPER)]},
                  open(os.path.join(DATA, "blade_bench.json"), "w"), indent=2)
