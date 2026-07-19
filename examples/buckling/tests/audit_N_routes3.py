"""audit_N_routes3.py -- WHY does the geometric route 2 lose up to 45% on thick spar-cap walls?
Diagnose the worst element: per through-thickness point, intended vs projected ring element,
intended depth vs the depth stress_at_points actually used, and the wall thickness vs element arc length."""
import os, sys, pickle
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

FF = [1.0e6, 3.0e5, 8.0e5, 2.0e6, 5.0e7, 1.2e7]

for i, ew in ((20, 84), (30, 144)):
    B = pickle.load(open(os.path.join(CACHE, "bundle_s%02d.pkl" % i), "rb"))
    ay = load_station_abd(os.path.join(IEA, "dehom51", "out", "abd", "iea_s%02d_abd.yaml" % i))
    abd = ay["by_name"]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
    layups = B["layup_per_elem"]; frac = float(B.get("frac", 0.0))
    L_e = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
    nm = layups[ew]; h = float(abd[nm][2])
    print("=" * 96)
    print("station %02d  element %d  layup %s :  wall h = %.1f mm ,  ring element arc length = %.1f mm"
          % (i, ew, nm, 1e3 * h, 1e3 * L_e[ew]))
    print("  h / arc-length = %.2f   (>1 means the through-thickness excursion exceeds the element size)"
          % (h / L_e[ew]))
    print("  station-wide: median arc %.1f mm ; walls thicker than their own element: %d / %d elements"
          % (1e3 * np.median(L_e), int(sum(1 for e in range(len(rc)) if float(abd[layups[e]][2]) > L_e[e])), len(rc)))

    c0, c1 = int(rc[ew, 0]), int(rc[ew, 1]); mid = 0.5 * (corners[c0] + corners[c1])
    t2, t3 = corners[c1] - corners[c0]; tl = np.hypot(t2, t3); t2, t3 = t2 / tl, t3 / tl
    n2, n3 = t3, -t2
    if (cen[0] - mid[0]) * n2 + (cen[1] - mid[1]) * n3 < 0:
        n2, n3 = -n2, -n3
    zs = np.linspace(0.0, h, 11)                       # OML -> IML
    pts = np.array([[mid[0] + (z - frac * h) * n2, mid[1] + (z - frac * h) * n3] for z in zs])
    res = dehom_rm.stress_at_points(B, pts, beam_force_vabs=FF, frame="global", n_per_layer=4)
    got = np.asarray(res["elem"]); dep = np.asarray(res["depth"]); S = np.asarray(res["stress"])
    print("   z_oml[mm]  intended_elem  got_elem  got_layup    z_used_ring[mm]  z_expected[mm]   sigma11[Pa]")
    for k in range(len(zs)):
        flag = "" if got[k] == ew else "   <-- MISPROJECTED"
        print("   %8.2f  %11d  %8d  %-10s %14.2f %14.2f  %+12.4e%s"
              % (1e3 * zs[k], ew, got[k], layups[got[k]], 1e3 * dep[k], 1e3 * (zs[k] - frac * h), S[k, 0], flag))
    # how close is the nearest OTHER part of the contour?
    d = np.hypot(corners[:, 0] - mid[0], corners[:, 1] - mid[1])
    order = np.argsort(d)
    far = [j for j in order if j not in (c0, c1)][:3]
    print("  nearest contour nodes to this element's midpoint (excluding its own): "
          + ", ".join("node %d at %.1f mm" % (j, 1e3 * d[j]) for j in far))
    print("  -> half-wall excursion = %.1f mm ; if that exceeds the distance to the opposite/adjacent wall,"
          % (1e3 * h / 2))
    print("     _project_point snaps the deep point onto THAT wall instead (wrong layup, wrong depth).")
