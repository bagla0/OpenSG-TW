"""Measure the through-thickness span of each st15 dehom path .coords file,
to pick the thinnest panel for the shell-vs-solid dehom comparison."""
import os, numpy as np
PDIR = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
        r"\opensg-FEniCS\data\st15_path_coords-20260614T203452Z-3-001\st15_path_coords")
paths = ["solid.lp_sparcap_center_thickness_015.coords",
         "solid.lp_fore_panel_thickness_015.coords",
         "solid.lp_aft_panel_thickness_015.coords",
         "solid.hp_fore_panel_thickness_015.coords",
         "solid.hp_aft_panel_thickness_015.coords",
         "solid.le_lp_reinf_thickness_015.coords",
         "solid.fore_web_thickness_015.coords",
         "solid.lp_sparcap_left_edge_thickness_015.coords"]
rows = []
for pf in paths:
    fp = os.path.join(PDIR, pf)
    if not os.path.exists(fp):
        print("missing", pf); continue
    c = np.loadtxt(fp)[:, :2]
    span = float(np.hypot(*(c[-1]-c[0])))
    # also the bounding extent (robust if path not monotone)
    ext = float(np.hypot(*(c.max(0)-c.min(0))))
    mid = c.mean(0)
    rows.append((pf.replace("solid.", "").replace("_thickness_015.coords", ""),
                 span*1e3, ext*1e3, len(c), mid))
rows.sort(key=lambda r: r[2])
print(f"{'path':28s}{'span(mm)':>10s}{'extent(mm)':>12s}{'npts':>6s}   mid(y2,y3)")
for nm, sp, ex, n, mid in rows:
    print(f"{nm:28s}{sp:10.2f}{ex:12.2f}{n:6d}   ({mid[0]:.3f},{mid[1]:.3f})")
