"""s12_recon.py -- diagnose the sigma12 web discrepancy RM-vs-VABS at r=0.2 (iea_s10).

Hypothesis test: if RM and VABS AGREE on the in-plane Mohr invariants
    I1 = s11+s22,   R = sqrt(((s11-s22)/2)^2 + s12^2)
at the web gauss points but DISAGREE on s12 itself, the discrepancy is a
MATERIAL-FRAME (ply-angle) convention difference, not a mechanics error.
Also dumps the web layup definitions from the OpenSG shell yaml and the VABS
.sg layer table, and reports the implied rotation angle per web point.
"""
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
VABS = os.path.join(IEA, "out", "VABS_iea51")
SHELL = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")

FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]
print("FF (VABS order F1 F2 F3 M1 M2 M3):", np.array2string(FF, precision=3))

# ---- VABS gauss stress ----
dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6      # Voigt [11,22,33,23,13,12] MPa
# ---- RM cached gauss stress (material frame, same points) ----
z = np.load(os.path.join(HERE, "_rm_s10_cache.npz"))
sRg = z["sRg"]
assert len(sRg) == len(sVg), (len(sRg), len(sVg))

# ---- web/skin split: distance from the OUTER boundary loop of the .sg mesh ----
def parse_sg(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    hi = next(i for i, l in enumerate(L) if len(l.split()) == 3
              and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne, nm = [int(x) for x in L[hi].split()]
    nodes = np.array([[float(x) for x in L[hi + 1 + k].split()[1:3]] for k in range(nn)])
    conn = [[int(x) for x in L[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
    # per-element layer/theta line follows connectivity in VABS .sg: "eid layer theta"
    lay = {}
    for k in range(ne):
        parts = L[hi + 1 + nn + ne + k].split()
        if len(parts) >= 3:
            lay[int(parts[0])] = (int(float(parts[1])), float(parts[2]))
    return nodes, conn, lay, L, hi, nn, ne, nm

nodes, conn, elay, Lraw, hi, nn, ne, nm = parse_sg(os.path.join(VABS, "iea_s10.sg"))
tris = []
for c in conn:
    c0 = [n - 1 for n in c]
    if len(c0) == 3:
        tris.append(c0[:3])
    elif len(c0) >= 4:
        tris.append([c0[0], c0[1], c0[2]]); tris.append([c0[0], c0[2], c0[3]])
tris = np.array(tris)
from collections import defaultdict
ec = defaultdict(int)
for t in tris:
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        ec[(min(a, b), max(a, b))] += 1
bed = [e for e, c in ec.items() if c == 1]
adj = defaultdict(list)
for a, b in bed:
    adj[a].append(b); adj[b].append(a)
loops, seen = [], set()
for s in adj:
    if s in seen:
        continue
    loop, cur, prev = [s], s, -1
    seen.add(s)
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        if cur == s:
            break
        loop.append(cur); seen.add(cur)
    loops.append(loop)
loops.sort(key=lambda l: -len(l))
outer = np.array(loops[0])
print("boundary loops:", [len(l) for l in loops[:6]], " outer loop nodes:", len(outer))
op = nodes[outer]
from scipy.spatial import cKDTree
kd = cKDTree(op)
d_out = kd.query(sm_xy)[0]
thr = 0.5 * np.percentile(d_out, 99)  # webs sit far from the outer loop
web = d_out > max(0.06, thr * 0.5)
print("web threshold %.3f m -> %d web gauss pts / %d total" % (max(0.06, thr * 0.5), web.sum(), len(web)))

def stats(name, mask):
    V, Rm = sVg[mask], sRg[mask]
    I1v, I1r = V[:, 0] + V[:, 1], Rm[:, 0] + Rm[:, 1]
    Rv = np.sqrt(((V[:, 0] - V[:, 1]) / 2) ** 2 + V[:, 5] ** 2)
    Rr = np.sqrt(((Rm[:, 0] - Rm[:, 1]) / 2) ** 2 + Rm[:, 5] ** 2)
    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    print("\n=== %s (%d pts) ===" % (name, mask.sum()))
    print(" rms s11   VABS %8.3f  RM %8.3f  MPa   rms diff %8.3f" % (rms(V[:, 0]), rms(Rm[:, 0]), rms(V[:, 0] - Rm[:, 0])))
    print(" rms s22   VABS %8.3f  RM %8.3f  MPa   rms diff %8.3f" % (rms(V[:, 1]), rms(Rm[:, 1]), rms(V[:, 1] - Rm[:, 1])))
    print(" rms s12   VABS %8.3f  RM %8.3f  MPa   rms diff %8.3f" % (rms(V[:, 5]), rms(Rm[:, 5]), rms(V[:, 5] - Rm[:, 5])))
    print(" rms I1    VABS %8.3f  RM %8.3f  MPa   rms diff %8.3f  <- frame-invariant" % (rms(I1v), rms(I1r), rms(I1v - I1r)))
    print(" rms MohrR VABS %8.3f  RM %8.3f  MPa   rms diff %8.3f  <- frame-invariant" % (rms(Rv), rms(Rr), rms(Rv - Rr)))
    big = mask.copy()
    big[mask] = Rv > np.percentile(Rv, 75)
    V2, R2 = sVg[big], sRg[big]
    phiV = 0.5 * np.arctan2(2 * V2[:, 5], V2[:, 0] - V2[:, 1])
    phiR = 0.5 * np.arctan2(2 * R2[:, 5], R2[:, 0] - R2[:, 1])
    dphi = np.rad2deg(np.unwrap(phiR - phiV))
    dphi = (np.rad2deg(phiR - phiV) + 90.0) % 180.0 - 90.0
    print(" implied frame rotation (deg, top-quartile MohrR): median %.1f  IQR [%.1f, %.1f]"
          % (np.median(dphi), np.percentile(dphi, 25), np.percentile(dphi, 75)))

stats("SKIN", ~web)
stats("WEB", web)

# ---- layup definitions: OpenSG yaml sections ----
d = yaml.safe_load(open(SHELL))
print("\n=== OpenSG shell yaml sections (name: [mat, thick, angle] plies) ===")
for s in d["sections"]:
    nm_ = s.get("elementSet", "?")
    lay = s.get("layup", [])
    ang = sorted(set(float(p[2]) for p in lay))
    print(" %-28s %2d plies  angles %s" % (nm_, len(lay), ang))

# ---- VABS .sg layer table at a few web vs skin elements ----
# element centroids -> web flag
cen_e, eid = [], []
k = 0
for c in conn:
    c0 = [n - 1 for n in c]
    cen_e.append(nodes[c0].mean(0)); eid.append(k + 1); k += 1
cen_e = np.array(cen_e)
d_e = kd.query(cen_e)[0]
webe = d_e > max(0.06, thr * 0.5)
print("\n=== VABS .sg element (layer,theta1) samples ===")
wl = [elay.get(i + 1) for i in np.where(webe)[0][:8]]
sl = [elay.get(i + 1) for i in np.where(~webe)[0][:8]]
print(" web  elements:", wl)
print(" skin elements:", sl)
# layer-id -> (material, theta3/ply angle) table lives after the element block in .sg;
# dump the first lines that follow for manual inspection
tail0 = hi + 1 + nn + 2 * ne
print("\n.sg lines after element/theta block (layer defs):")
for l in Lraw[tail0:tail0 + 12]:
    print("   ", l)
