"""audit_N_routes2.py -- part 2 of the N-recovery audit.

(A) ALL ring elements, GENERAL 6-component load, all 3 components of N: max|ratio-1| route2/route1.
(B) The PRODUCTION geometric route 2: dehom_rm.stress_at_points with points placed along the wall
    normal (so it goes through _project_point).  Does the projection put deep-wall points on the
    RIGHT element?  This is the geometric robustness question route 1 never faces.
(C) Batched cost of the production geometric route 2 for a full station.
"""
import os, sys, time, pickle
import numpy as np

ROOT = "/home/roger/a/bagla0/OpenSG-TW-claude"
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")
IEA = os.path.join(ROOT, "examples", "data", "iea_all_stations")
BUCK = os.path.join(ROOT, "examples", "buckling")
CACHE = os.path.join(BUCK, "data", "blade_cache")
sys.path.insert(0, ROOT); sys.path.insert(0, XSEC); sys.path.insert(0, BUCK)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from emit_abd import load_station_abd
from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, plate_stress_at_depth

# general load: axial + both shears + torsion + both moments (VABS order F1,F2,F3,M1,M2,M3)
FF = [1.0e6, 3.0e5, 8.0e5, 2.0e6, 5.0e7, 1.2e7]
STATIONS = [5, 10, 20, 30, 40]


def bundle(i):
    return pickle.load(open(os.path.join(CACHE, "bundle_s%02d.pkl" % i), "rb"))


def abd_path(i):
    return os.path.join(IEA, "dehom51", "out", "abd", "iea_s%02d_abd.yaml" % i)


def warp_for(B, name):
    inf = B["layup_db"][name]
    return compute_ABD_matrix(inf["thick"], inf["angles"], inf["mat_names"], B["material_db"],
                              n_per_layer=4, return_warping=True, elem_order=2)[2]


def ply_z(plies, ng=2):
    """ply-wise Gauss depths z (from OML) + weights, exact for the piecewise stress field."""
    xg, wg = np.polynomial.legendre.leggauss(ng)
    z = []; w = []; z0 = 0.0
    for p in plies:
        t = float(p["thickness"]); z1 = z0 + t
        z += list(0.5 * (z1 - z0) * xg + 0.5 * (z1 + z0)); w += list(0.5 * t * wg)
        z0 = z1
    return np.array(z), np.array(w)


print("=" * 100)
print("(A) ALL elements, GENERAL load FF =", FF)
print("    route2 = plywise-exact int sigma dz (plate level) ; route1 = A@s6[:3]+B@s6[3:]")
print("=" * 100)
print("  %-5s %6s | %-28s %-28s %-28s" % ("sta", "nelem", "max|N11r-1|", "max|N22r-1|", "max|N12r-1|"))
for i in STATIONS:
    B = bundle(i)
    ay = load_station_abd(abd_path(i)); abd = ay["by_name"]
    raw = {L["name"]: L for L in ay["raw"]["layups"]}
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
    rc = np.asarray(B["red_cells"]); layups = B["layup_per_elem"]; frac = float(B.get("frac", 0.0))
    Wc = {nm: warp_for(B, nm) for nm in set(layups)}
    Zc = {nm: ply_z(raw[nm]["plies"]) for nm in set(layups)}
    d1 = np.zeros((len(rc), 3)); d2 = np.zeros((len(rc), 3))
    for e in range(len(rc)):
        nm = layups[e]
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        A6 = np.asarray(abd[nm][0])
        d1[e] = A6[:3, :3] @ s6[:3] + A6[:3, 3:] @ s6[3:6]
        w = Wc[nm]; h = float(w["node_x"][-1])
        s6r = np.array(s6, float); s6r[0:3] -= frac * h * s6r[3:6]
        zz, ww = Zc[nm]
        S = np.array([np.asarray(plate_stress_at_depth(w, s6r, float(z))[1], float) for z in zz])
        Sg = (ww[:, None] * S).sum(0)
        d2[e] = [Sg[0], Sg[1], Sg[5]]
    r = []
    for k in range(3):
        sc = np.abs(d1[:, k]).max()
        r.append(np.abs(d2[:, k] - d1[:, k]).max() / (sc + 1e-30))     # normalised by the field max
    print("  s%02d   %6d | %.3e (rel to max)      %.3e                   %.3e" % (i, len(rc), r[0], r[1], r[2]))

print("\n" + "=" * 100)
print("(B) PRODUCTION geometric route 2 (dehom_rm.stress_at_points, points on the wall normal)")
print("    -> does _project_point land deep-wall points on the intended ring element?")
print("=" * 100)
print("  %-5s %7s %8s %10s %10s | %s" % ("sta", "npts", "misproj", "ratio_med", "ratio_max", "worst-elem detail"))
for i in STATIONS:
    B = bundle(i)
    ay = load_station_abd(abd_path(i)); abd = ay["by_name"]
    raw = {L["name"]: L for L in ay["raw"]["layups"]}
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    cen = corners.mean(0); layups = B["layup_per_elem"]; frac = float(B.get("frac", 0.0))
    Zc = {nm: ply_z(raw[nm]["plies"]) for nm in set(layups)}
    pts = []; owner = []; wts = []
    for e in range(len(rc)):
        nm = layups[e]; h = float(abd[nm][2])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1]); mid = 0.5 * (corners[c0] + corners[c1])
        t2, t3 = corners[c1] - corners[c0]; tl = np.hypot(t2, t3); t2, t3 = t2 / tl, t3 / tl
        n2, n3 = t3, -t2
        if (cen[0] - mid[0]) * n2 + (cen[1] - mid[1]) * n3 < 0:
            n2, n3 = -n2, -n3                                   # inward normal
        zz, ww = Zc[nm]
        for z, wq in zip(zz, ww):
            zr = z - frac * h                                    # OML depth -> ring-ref depth
            pts.append([mid[0] + zr * n2, mid[1] + zr * n3]); owner.append(e); wts.append(wq)
    pts = np.array(pts); owner = np.array(owner); wts = np.array(wts)
    t0 = time.time()
    res = dehom_rm.stress_at_points(B, pts, beam_force_vabs=FF, frame="global", n_per_layer=4)
    tgeo = time.time() - t0
    got = np.asarray(res["elem"]); S = np.asarray(res["stress"])
    mis = int((got != owner).sum())
    Ng = np.zeros(len(rc))
    for k in range(len(pts)):
        Ng[owner[k]] += wts[k] * S[k, 0]
    Nd = np.zeros(len(rc))
    for e in range(len(rc)):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        A6 = np.asarray(abd[layups[e]][0])
        Nd[e] = (A6[:3, :3] @ s6[:3] + A6[:3, 3:] @ s6[3:6])[0]
    sc = np.abs(Nd).max()
    rat = np.where(np.abs(Nd) > 0.01 * sc, Ng / np.where(np.abs(Nd) > 0, Nd, 1), np.nan)
    err = np.abs(Ng - Nd) / sc
    ew = int(np.nanargmax(err))
    print("  s%02d   %7d %7d%%  %10.4f %10.4f | elem %d (%s): direct %+.3e geo %+.3e (err %.1f%% of max) [%.2fs]"
          % (i, len(pts), round(100 * mis / len(pts)), np.nanmedian(rat), np.nanmax(np.abs(rat)),
             ew, layups[ew], Nd[ew], Ng[ew], 100 * err[ew], tgeo))
    print("        misprojected points: %d/%d ; ||Ngeo-Ndirect||/||Ndirect|| = %.4f ; max elem err = %.2f%% of field max"
          % (mis, len(pts), np.linalg.norm(Ng - Nd) / np.linalg.norm(Nd), 100 * err.max()))
    print("        batched geometric route-2 cost: %.2f s / station -> %.0f s blade-wide (51 stations)"
          % (tgeo, 51 * tgeo))
