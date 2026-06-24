"""Emit a solid+shell orientation PNG per station into validation/orient_stations/, AND run a numerical
e2/e3 consistency check (web + TE-skin, shell vs solid) so any inconsistent station is flagged."""
import os, sys
import numpy as np, yaml
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section
from fe_jax.orient_plot import plot_orient

VAL = os.path.join(CC, "windio_converter", "validation")
OUT = os.path.join(VAL, "orient_stations"); os.makedirs(OUT, exist_ok=True)
blade = WindIOBlade(os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml"))
STATIONS = [(round(0.1 * k, 2), "r%03d" % (10 * k)) for k in range(1, 10)] + [(0.95, "r095")]


def load(p):
    d = yaml.safe_load(open(p))
    nd = np.array([[float(v) for v in str(r[0]).split()][:2] for r in d["nodes"]])
    el = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    ori = np.array([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in d["elementOrientations"]])
    return nd, el, ori


def web_e3(nodes, elems, oris, is_solid, xc, bw):
    e2s, e3s = [], []
    for k, e in enumerate(elems):
        cen = nodes[e[:3 if is_solid else 2]].mean(axis=0); e2 = oris[k, 3:5]
        if abs(e2[1]) > 0.85 and xc - bw <= cen[0] <= xc + bw:
            sgn = 1.0 if e2[1] > 0 else -1.0
            e2s.append(sgn * e2); e3s.append(sgn * oris[k, 6:8])
    return (np.mean(e2s, 0), np.mean(e3s, 0)) if e2s else (None, None)


def skin_te_e3(nodes, elems, oris, is_solid, xmin, ysign):
    acc = np.zeros(2); n = 0
    for k, e in enumerate(elems):
        cen = nodes[e[:3 if is_solid else 2]].mean(axis=0)
        if cen[0] > xmin and ysign * cen[1] > 0:
            e3 = oris[k, 6:8].copy()
            if ysign * e3[1] > 0:
                e3 = -e3
            acc += e3; n += 1
    return acc / n if n else None


print("  r    | web dot(e2) dot(e3) | TE-skin dot(upper) dot(lower) | verdict")
for r, tag in STATIONS:
    sh = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    so = os.path.join(VAL, "solid_iea22_%s.yaml" % tag)
    has_solid = os.path.exists(so)
    plot_orient(sh, so if has_solid else None, os.path.join(OUT, "orient_iea22_%s.png" % tag))
    if not has_solid:
        print("  %.2f | (no solid yaml)" % r); continue
    cs = build_cross_section(blade, r, mesh_size=0.01); chord = cs["chord"]
    sn, se, so_ = load(sh); qn, qe, qo = load(so)
    wx = sorted({round(float(cs["nodes"][w["a"]][0]), 3) for w in cs["webs"]})
    we2, we3 = [], []
    for x in wx:
        a = web_e3(sn, se, so_, False, x, 0.04 * chord); b = web_e3(qn, qe, qo, True, x, 0.04 * chord)
        if a[0] is not None and b[0] is not None:
            we2.append(np.dot(a[0], b[0])); we3.append(np.dot(a[1], b[1]))
    de2 = float(np.mean(we2)) if we2 else float("nan"); de3 = float(np.mean(we3)) if we3 else float("nan")
    su = skin_te_e3(sn, se, so_, False, 0.85 * chord, +1); qu = skin_te_e3(qn, qe, qo, True, 0.85 * chord, +1)
    sl = skin_te_e3(sn, se, so_, False, 0.85 * chord, -1); ql = skin_te_e3(qn, qe, qo, True, 0.85 * chord, -1)
    du = float(np.dot(su / np.linalg.norm(su), qu / np.linalg.norm(qu))) if su is not None and qu is not None else float("nan")
    dl = float(np.dot(sl / np.linalg.norm(sl), ql / np.linalg.norm(ql))) if sl is not None and ql is not None else float("nan")
    ok = (de2 > 0.7 and de3 > 0.7 and (du > 0.7 or np.isnan(du)) and (dl > 0.7 or np.isnan(dl)))
    print("  %.2f |   %+.3f   %+.3f  |     %+.3f       %+.3f      | %s"
          % (r, de2, de3, du, dl, "OK" if ok else "*** CHECK ***"))
print("\nwrote per-station PNGs -> validation/orient_stations/")
