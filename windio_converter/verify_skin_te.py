"""Verify the SKIN e3 (OML->IML) AGREES between 1D-shell and 2D-solid, focusing on the TE region where the
old centroid heuristic flipped. Compares the mean skin e3 over upper-TE and lower-TE bands, and reports the
fraction of ALL shell skin elements whose e3 points consistently inward (matching the solid skin)."""
import os, sys
import numpy as np, yaml
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "windio_converter"))
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section

VAL = os.path.join(CC, "windio_converter", "validation")
blade = WindIOBlade(os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml"))


def load(p):
    d = yaml.safe_load(open(p))
    nodes = np.array([[float(v) for v in str(r[0]).split()][:2] for r in d["nodes"]])
    elems = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    oris = np.array([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in d["elementOrientations"]])
    sets = {s["name"]: set(int(x) - 1 for x in s["labels"]) for s in d["sets"]["element"]}
    return nodes, elems, oris, sets


def mean_skin_e3(nodes, elems, oris, is_solid, xmin, ysign):
    acc = np.zeros(2); n = 0
    for k, e in enumerate(elems):
        cen = nodes[e[:3 if is_solid else 2]].mean(axis=0)
        if cen[0] > xmin and ysign * cen[1] > 0:
            e3 = oris[k, 6:8].copy()
            if ysign * e3[1] > 0:                          # align to inward (upper inward = -y, lower = +y)
                e3 = -e3
            acc += e3; n += 1
    return (acc / n if n else acc), n


cs = build_cross_section(blade, 0.5, mesh_size=0.01); chord = cs["chord"]
sn, se, so, ssets = load(os.path.join(VAL, "shell_iea22_r050.yaml"))
qn, qe, qo, _ = load(os.path.join(VAL, "solid_iea22_r050.yaml"))
web_ids = set().union(*[ssets[n] for n in ssets if n == "layup_5"]) if "layup_5" in ssets else set()
xmin = 0.85 * chord
print("=== IEA-22 r050 : SKIN e3 (OML->IML) near TE, 1D-shell vs 2D-solid (x > %.2f m) ===" % xmin)
for side, ys in (("upper-TE", +1), ("lower-TE", -1)):
    s_e3, ns = mean_skin_e3(sn, se, so, False, xmin, ys)
    q_e3, nq = mean_skin_e3(qn, qe, qo, True, xmin, ys)
    d = float(np.dot(s_e3 / (np.linalg.norm(s_e3) + 1e-9), q_e3 / (np.linalg.norm(q_e3) + 1e-9)))
    print("  %-9s shell e3=(%+.2f,%+.2f) [%d]  solid e3=(%+.2f,%+.2f) [%d]  dot=%+.3f -> %s"
          % (side, s_e3[0], s_e3[1], ns, q_e3[0], q_e3[1], nq, d, "AGREE" if d > 0.7 else "MISMATCH"))

# consistency: every shell skin element's e3 should point inward (toward the section interior)
C = sn.mean(axis=0); flipped = 0; tot = 0
for k, e in enumerate(se):
    if k in web_ids:
        continue
    cen = sn[e[:2]].mean(axis=0); e3 = so[k, 6:8]
    tot += 1
    if np.dot(e3, C - cen) < 0:                            # points away from interior
        flipped += 1
print("  shell skin elements with e3 pointing OUTWARD (should be ~0): %d / %d" % (flipped, tot))
