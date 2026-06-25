"""Replicate the REFERENCE mh104 1D-shell mesh EXACTLY (mh104_0.2thickTimo_FEniCS.py.py): read
datapoints_mh104.txt (LE-start, dividers labelled), segment counter + symmetric remap, FLOATING
3-node webs (top, chord-midpoint, bottom) -- NO resampling, NO connected webs.  Emit it as a YAML
so we can run it through ShellBounMesh and see if the reference mesh reproduces 5.52e8 EA (isolating
my generator's resampling/connected-web from the ShellBounMesh solver)."""
import os
import sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DP = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data\mh104_training\prevabs to ymal\datapoints_mh104.txt"
F = next((float(a[2:]) for a in sys.argv if a.startswith("f=")), 0.2)   # ply-thickness factor, e.g. f=0.4
CONNECT = ("connect" in sys.argv)   # connect web to nearest skin node vs floating
REVERSE = ("cw" not in sys.argv)    # DEFAULT now CCW: traverse the contour to match the XML baseline flow
                                    # (l34:l23 -> top TE->LE) so e2 matches the VABS-validated solid and off-axis
                                    # plies are NOT mirror-imaged.  Pass "cw" to force the old datapoint-order (CW).
OUT = os.path.join(HERE, "shell_ref_f%03d%s%s.yaml" % (round(F * 100), "_connect" if CONNECT else "", "" if REVERSE else "_cw"))

MAT = {
    "gelcoat": dict(E=[10.0, 10.0, 10.0], G=[1.0, 1.0, 1.0], nu=[0.3, 0.3, 0.3], rho=1830.0),
    "nexus": dict(E=[1.03e10, 1.03e10, 1.03e10], G=[8e9, 8e9, 8e9], nu=[0.3, 0.3, 0.3], rho=1664.0),
    "db_frp": dict(E=[1.03e10, 1.03e10, 1.03e10], G=[8e9, 8e9, 8e9], nu=[0.3, 0.3, 0.3], rho=1830.0),
    "ud_frp": dict(E=[3.7e10, 9e9, 9e9], G=[4e9, 4e9, 4e9], nu=[0.28, 0.28, 0.28], rho=1860.0),
    "balsa": dict(E=[1e7, 1e7, 1e7], G=[2e5, 2e5, 2e5], nu=[0.3, 0.3, 0.3], rho=128.0),
}
TH = dict(gelcoat=0.000381, nexus=0.00051, db_frp=0.00053, ud_frp=0.00053, balsa=0.003125)


def pl(m, n, a):
    return [m, round(TH[m] * n * F, 10), a]


# sections in sub order 0..4 (datapoints LE-start remap): 0=LE,1=midFwd,2=spar,3=TE,4=web
LAYUPS = [
    [pl("gelcoat", 1, 0), pl("nexus", 1, 0), pl("db_frp", 18, 20)],
    [pl("gelcoat", 1, 0), pl("nexus", 1, 0), pl("db_frp", 33, 20)],
    [pl("gelcoat", 1, 0), pl("nexus", 1, 0), pl("db_frp", 17, 20), pl("ud_frp", 38, 30), pl("balsa", 1, 0), pl("ud_frp", 37, 30), pl("db_frp", 16, 20)],
    [pl("gelcoat", 1, 0), pl("nexus", 1, 0), pl("db_frp", 17, 20), pl("balsa", 1, 0), pl("db_frp", 16, 0)],
    [pl("ud_frp", 38, 0), pl("balsa", 1, 0), pl("ud_frp", 38, 0)],
]


class FlowList(list):
    pass


yaml.add_representer(FlowList, lambda d, data: d.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))

dd = ["l12", "l23", "l34", "h34", "h23", "h12"]
rows = [l for l in open(DP).read().splitlines() if l.strip()]
sub, c, pts = [], 0, []
for line in rows:
    p = line.split()
    if p[0] in dd:
        c += 1
    sub.append(c)
    pts.append([1.9 * (float(p[1]) - 0.25), 1.9 * float(p[2])])   # model [X, Y]
sub = np.array(sub, dtype=int)
sub[sub == 4] = 2; sub[sub == 5] = 1; sub[sub == 6] = 0
pts = np.array(pts)
if REVERSE:
    pts = pts[::-1].copy(); sub = sub[::-1].copy()   # CCW traversal -> e2 follows the XML baseline direction
nnode = len(pts)
print("datapoints:", nnode, " sub layups:", sorted(set(sub.tolist())), " traversal:", "CCW" if REVERSE else "CW")

# floating webs (reference logic)
w1, w2 = 1.9 * (0.161 - 0.25), 1.9 * (0.511 - 0.25)
y1, y2 = [], []
for i in range(nnode - 1):
    a, b = pts[i], pts[i + 1]
    if (w1 >= a[0] and w1 <= b[0]) or (w1 <= a[0] and w1 >= b[0]):
        y1.append([w1, a[1] + (b[1] - a[1]) * ((w1 - a[0]) / (b[0] - a[0]))])
    if (w2 >= a[0] and w2 <= b[0]) or (w2 <= a[0] and w2 >= b[0]):
        y2.append([w2, a[1] + (b[1] - a[1]) * ((w2 - a[0]) / (b[0] - a[0]))])
nphases = int(sub.max())
if not CONNECT:
    # FLOATING webs (reference): 3 separate nodes per web, NOT shared with the skin
    y1 = np.array([y1[0], [w1, 0.0], y1[1]]); y2 = np.array([y2[0], [w2, 0.0], y2[1]])
    elem = [[k, k + 1] for k in range(nnode - 1)]; elem.append([nnode - 1, 0])
    subL = list(sub)
    pts = np.concatenate([pts, y1, y2])
    for web in range(4):
        nn = nnode + 2 if web < 2 else nnode + 7
        elem.append([nn - web, nn - web - 1]); subL.append(nphases + 1)
    pts = np.array(pts); elem = np.array(elem); subL = np.array(subL)
else:
    # STRICT CONNECTED webs: insert each web crossing as a SHARED skin node (splitting the host skin
    # element) so the web is continuous with the skin -- exactly like the 2D solid mesh.
    nodes = [list(p) for p in pts]; subN = list(sub)          # per-NODE layup tag (for the inserted crossings)
    elem = [[k, k + 1] for k in range(nnode - 1)]; elem.append([nnode - 1, 0])
    esub = list(sub[:nnode - 1]) + [int(sub[nnode - 1])]      # per-ELEMENT layup
    webpairs = []
    for wx in (w1, w2):
        cross = []
        for ei in range(len(elem)):                           # find the 2 skin elements the vertical X=wx crosses
            a, b = elem[ei]; xa, xb = nodes[a][0], nodes[b][0]
            if (xa - wx) * (xb - wx) < 0:
                t = (wx - xa) / (xb - xa); yy = nodes[a][1] + t * (nodes[b][1] - nodes[a][1])
                cross.append((ei, [wx, yy], esub[ei]))
        cross = cross[:2]
        ins = []
        for (ei, P, es) in sorted(cross, key=lambda c: -c[0]):  # insert from high ei so indices stay valid
            nid = len(nodes); nodes.append(P); subN.append(es)
            a, b = elem[ei]; elem[ei] = [a, nid]; elem.insert(ei + 1, [nid, b]); esub.insert(ei + 1, es)
            ins.append((nid, P[1]))
        webpairs.append(sorted(ins, key=lambda z: z[1]))        # (bottom, top) by Y
    for (nb, _), (nt, _) in webpairs:
        Pb, Pt = nodes[nb], nodes[nt]; mid = len(nodes); nodes.append([Pb[0], 0.0])
        elem.append([nb, mid]); esub.append(nphases + 1)
        elem.append([mid, nt]); esub.append(nphases + 1)
    pts = np.array(nodes); elem = np.array(elem); subL = np.array(esub)
skin = pts
print("nodes:", len(pts), " elements:", len(elem), " web sub:", nphases + 1)

# orientations: e1=beam(z), e2=tangent, e3=ply-normal.  SKIN: flip e3 inward (OML->IML).
# WEB: keep the natural e3 = e1 x e2 (= -x for both webs, built bottom->top), matching the SOLID's
# consistent web ply-stacking; do NOT flip toward the centroid (that wrongly reverses web1's e3).
C = pts.mean(axis=0)
wsub = int(subL.max())
oris = []
for i, (a, b) in enumerate(elem):
    P1, P2 = pts[a], pts[b]
    t = P2 - P1; e2 = t / (np.linalg.norm(t) + 1e-30); e3 = np.array([-e2[1], e2[0]])
    if subL[i] != wsub and np.dot(e3, C - 0.5 * (P1 + P2)) < 0:
        e3 = -e3
    oris.append([0.0, 0.0, 1.0, float(e2[0]), float(e2[1]), 0.0, float(e3[0]), float(e3[1]), 0.0])

seg = {"nodes": [], "elements": [], "sets": {"element": []}, "sections": [], "elementOrientations": [], "materials": []}
for P in pts:
    seg["nodes"].append(FlowList(["%.8f %.8f 0.0" % (P[0], P[1])]))
for (a, b) in elem:
    seg["elements"].append(FlowList(["%d %d" % (a + 1, b + 1)]))
for k in range(5):
    labels = [i + 1 for i in range(len(elem)) if subL[i] == k]
    seg["sets"]["element"].append({"name": "layup_%d" % k, "labels": labels})
for k in range(5):
    seg["sections"].append({"type": "shell", "elementSet": "layup_%d" % k,
                            "layup": [[p[0], float(p[1]), float(p[2])] for p in LAYUPS[k]]})
for o in oris:
    seg["elementOrientations"].append(FlowList([float(v) for v in o]))
used = []
for k in range(5):
    for p in LAYUPS[k]:
        if p[0] not in used:
            used.append(p[0])
for nm in used:
    m = MAT[nm]
    seg["materials"].append({"name": nm, "density": m["rho"],
                             "elastic": {"E": list(map(float, m["E"])), "G": list(map(float, m["G"])), "nu": list(map(float, m["nu"]))}})
yaml.dump(seg, open(OUT, "w"), sort_keys=False, default_flow_style=False)
print("wrote", OUT)
for k in range(5):
    print("  layup_%d  %d elems  %d plies" % (k, int((subL == k).sum()), len(LAYUPS[k])))
