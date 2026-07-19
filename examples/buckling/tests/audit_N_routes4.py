"""audit_N_routes4.py -- (i) decompose the original probe's 0.818 sandwich ratio into its two causes,
(ii) test 'bug 3' on the PRODUCTION buckling assembler (blade_buckling.station_abd conformal mesh +
KDTree layup lookup), which is what route 1 actually uses blade-wide."""
import os, sys, pickle, time
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

FF = [1.0e6, 0.0, 0.0, 0.0, 5.0e7, 0.0]          # the ORIGINAL probe's load


def warp_for(B, name):
    inf = B["layup_db"][name]
    return compute_ABD_matrix(inf["thick"], inf["angles"], inf["mat_names"], B["material_db"],
                              n_per_layer=4, return_warping=True, elem_order=2)[2]


print("=" * 100)
print("(i) decomposition of the ORIGINAL probe's sandwich ratio (station 10, thick-core layups)")
print("    plate-plywise = exact quadrature, no projection ; plate-41 = uniform trapz, no projection")
print("    geo-41 = the original probe (uniform trapz + _project_point) ; geo-plywise = exact quad + projection")
print("=" * 100)
i = 10
B = pickle.load(open(os.path.join(CACHE, "bundle_s%02d.pkl" % i), "rb"))
ay = load_station_abd(os.path.join(IEA, "dehom51", "out", "abd", "iea_s%02d_abd.yaml" % i))
abd = ay["by_name"]; raw = {L["name"]: L for L in ay["raw"]["layups"]}
st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
layups = B["layup_per_elem"]; frac = float(B.get("frac", 0.0))
L_e = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)

print("  %-6s %-9s %7s %7s %6s | %10s %10s %10s %10s" %
      ("elem", "layup", "h[mm]", "arc[mm]", "h/arc", "plate-ply", "plate-41", "geo-41", "geo-ply"))
for e in [0, 25, 55, 90, 130]:
    nm = layups[e]; h = float(abd[nm][2])
    s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    A6 = np.asarray(abd[nm][0])
    Nd = (A6[:3, :3] @ s6[:3] + A6[:3, 3:] @ s6[3:6])[0]
    w = warp_for(B, nm); hh = float(w["node_x"][-1])
    s6r = np.array(s6, float); s6r[0:3] -= frac * hh * s6r[3:6]
    # plate-level plywise exact
    xg, wg = np.polynomial.legendre.leggauss(4)
    acc = 0.0; z0 = 0.0
    for p in raw[nm]["plies"]:
        t = float(p["thickness"]); z1 = z0 + t
        for xq, wq in zip(xg, wg):
            z = 0.5 * (z1 - z0) * xq + 0.5 * (z1 + z0)
            acc += wq * 0.5 * t * plate_stress_at_depth(w, s6r, float(z))[1][0]
        z0 = z1
    Npp = acc
    # plate-level uniform 41
    zs = np.linspace(0.0, hh, 41)
    Np41 = np.trapz([plate_stress_at_depth(w, s6r, float(z))[1][0] for z in zs], zs)
    # geometric variants (through _project_point), same point sets
    c0, c1 = int(rc[e, 0]), int(rc[e, 1]); mid = 0.5 * (corners[c0] + corners[c1])
    t2, t3 = corners[c1] - corners[c0]; tl = np.hypot(t2, t3); t2, t3 = t2 / tl, t3 / tl
    n2, n3 = t3, -t2
    if (cen[0] - mid[0]) * n2 + (cen[1] - mid[1]) * n3 < 0:
        n2, n3 = -n2, -n3
    zc = np.linspace(-h / 2, h / 2, 41)
    pg = np.array([[mid[0] + z * n2, mid[1] + z * n3] for z in zc])
    Sg = np.asarray(dehom_rm.stress_at_points(B, pg, beam_force_vabs=FF, frame="global", n_per_layer=4)["stress"])
    Ng41 = np.trapz(Sg[:, 0], zc)
    zz = []; ww = []; z0 = 0.0
    for p in raw[nm]["plies"]:
        t = float(p["thickness"]); z1 = z0 + t
        zz += list(0.5 * (z1 - z0) * xg + 0.5 * (z1 + z0)); ww += list(0.5 * t * wg); z0 = z1
    zz = np.array(zz); ww = np.array(ww)
    pg2 = np.array([[mid[0] + (z - frac * h) * n2, mid[1] + (z - frac * h) * n3] for z in zz])
    Sg2 = np.asarray(dehom_rm.stress_at_points(B, pg2, beam_force_vabs=FF, frame="global", n_per_layer=4)["stress"])
    Ngp = float((ww * Sg2[:, 0]).sum())
    print("  %-6d %-9s %7.1f %7.1f %6.2f | %10.4f %10.4f %10.4f %10.4f"
          % (e, nm, 1e3 * h, 1e3 * L_e[e], h / L_e[e], Npp / Nd, Np41 / Nd, Ng41 / Nd, Ngp / Nd))

print("\n" + "=" * 100)
print("(ii) 'bug 3' on the PRODUCTION buckling assembler: blade_buckling.station_abd")
print("     (conformal N=%d-node loft + KDTree layup lookup) -- int A11 ds vs EA = C6[0,0]" % 120)
print("=" * 100)
import blade_buckling as bb
print("  %-6s %13s %13s %13s | %8s %8s %9s" %
      ("sta", "EA=C6[0,0]", "int A11 ds", "int A11red ds", "raw/EA", "red/EA", "perim[m]"))
for i in [5, 10, 20, 30, 40]:
    P, ABD, Gs = bb.station_abd(i)
    Bb = pickle.load(open(os.path.join(CACHE, "bundle_s%02d.pkl" % i), "rb"))
    EA = float(np.asarray(Bb["Timo"])[0, 0])
    mids0 = P[bb.sec_elems[:, 0]]; mids1 = P[bb.sec_elems[:, 1]]
    Ls = np.linalg.norm(mids1 - mids0, axis=1)
    iA = float(sum(ABD[e][0, 0] * Ls[e] for e in range(len(Ls))))
    iAr = float(sum((ABD[e][0, 0] - ABD[e][0, 1] ** 2 / ABD[e][1, 1]) * Ls[e] for e in range(len(Ls))))
    print("  s%02d    %.6e %.6e %.6e | %8.4f %8.4f %9.3f"
          % (i, EA, iA, iAr, iA / EA, iAr / EA, float(Ls.sum())))
