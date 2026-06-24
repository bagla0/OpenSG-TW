"""Numerically verify the WEB e2/e3 frames AGREE between the 1D-shell YAML and the 2D-solid YAML.
For each web (located by its chordwise x), gather near-vertical elements in a narrow x-band from both
meshes and compare the mean e2 and e3 (dot product ~ +1 => same direction)."""
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
    return nodes, elems, oris


def web_frames(nodes, elems, oris, is_solid, xband):
    """mean e2,e3 over near-vertical elements within [xband-w, xband+w]."""
    lo, hi = xband
    e2s, e3s = [], []
    for k, e in enumerate(elems):
        cen = nodes[e[:3 if is_solid else 2]].mean(axis=0)
        e2 = oris[k, 3:5]; e3 = oris[k, 6:8]
        if abs(e2[1]) > 0.85 and lo <= cen[0] <= hi:           # near-vertical + in band
            ref = e2 if e2[1] > 0 else -e2                     # sign-align e2 to +y3 (the chosen web sense)
            sgn = 1.0 if e2[1] > 0 else -1.0
            e2s.append(ref); e3s.append(sgn * e3)
    if not e2s:
        return None
    return np.mean(e2s, axis=0), np.mean(e3s, axis=0), len(e2s)


for r, tag in ((0.5, "r050"),):
    cs = build_cross_section(blade, r, mesh_size=0.01)
    chord = cs["chord"]
    sn, se, so = load(os.path.join(VAL, "shell_iea22_%s.yaml" % tag))
    qn, qe, qo = load(os.path.join(VAL, "solid_iea22_%s.yaml" % tag))
    # web chordwise x positions (from the shell-side attachment nodes)
    wx = sorted({round(float(cs["nodes"][w["a"]][0]), 4) for w in cs["webs"]})
    print("=== IEA-22 %s  web e2/e3 : 1D-shell vs 2D-solid (e2 sign-aligned to +y3) ===" % tag)
    for x in wx:
        bw = 0.04 * chord
        sh = web_frames(sn, se, so, False, (x - bw, x + bw))
        so_ = web_frames(qn, qe, qo, True, (x - bw, x + bw))
        if sh is None or so_ is None:
            print("  web x=%.2f : no elements found (shell=%s solid=%s)" % (x, sh is not None, so_ is not None)); continue
        e2s, e3s, ns = sh; e2q, e3q, nq = so_
        print("  web x=%.2f m  (shell %d / solid %d elems)" % (x, ns, nq))
        print("     shell  e2=(%+.2f,%+.2f) e3=(%+.2f,%+.2f)" % (e2s[0], e2s[1], e3s[0], e3s[1]))
        print("     solid  e2=(%+.2f,%+.2f) e3=(%+.2f,%+.2f)" % (e2q[0], e2q[1], e3q[0], e3q[1]))
        print("     dot(e2)=%+.3f  dot(e3)=%+.3f   -> %s"
              % (np.dot(e2s, e2q), np.dot(e3s, e3q),
                 "AGREE" if (np.dot(e2s, e2q) > 0.7 and np.dot(e3s, e3q) > 0.7) else "MISMATCH"))
