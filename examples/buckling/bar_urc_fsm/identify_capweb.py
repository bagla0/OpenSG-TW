"""identify_capweb.py -- is the mode-carrying layup_6 really the SPAR CAP, and does it lie BETWEEN the webs?

The mode at st06 localizes entirely in layup_6 (max amp 0.984 vs <=0.072 everywhere else). Before claiming
that reproduces the benchmark "spar cap between webs" mode, verify from the yaml itself:
  * what layup_6 is made of (a spar cap is a thick carbon_uni run; a panel is thin glass/foam)
  * where the shear webs attach (valence-3 nodes = web/skin T-junctions)
  * whether the mode elements sit between those attachment points on the compression side
"""
import os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ST = int(os.environ.get("ST", "6"))
shell = os.path.join(ROOT, "tests", "data", "1Dshell_%d.yaml" % ST)


def _row(x):
    out = []
    for v in ([x] if isinstance(x, str) else x):
        if isinstance(v, str):
            out.extend(float(t) for t in v.split())
        else:
            out.append(float(v))
    return out


d = yaml.safe_load(open(shell))
nd = np.array([_row(n)[:2] for n in d["nodes"]])
cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]]); cells -= cells.min()
name_of = {}
for grp in d["sets"]["element"]:
    for lab in grp["labels"]:
        name_of[int(lab) - 1] = grp["name"]
names = [name_of.get(k, d["sections"][0]["elementSet"]) for k in range(len(cells))]
lay = {s["elementSet"]: s["layup"] for s in d["sections"]}

print("BAR-URC station %d -- layup composition (total thickness, plies)\n" % ST)
print("   layup          t_tot [mm]   plies (material, t_mm, angle)")
for nm in sorted(lay, key=lambda s: -sum(float(p[1]) for p in lay[s])):
    pl = lay[nm]
    tt = sum(float(p[1]) for p in pl)
    desc = " | ".join("%s %.1f@%g" % (p[0], 1e3 * float(p[1]), float(p[2])) for p in pl)
    star = "  <== MODE LIVES HERE" if nm == "layup_6" else ""
    print("   %-12s   %8.2f     %s%s" % (nm, 1e3 * tt, desc, star))

# ---- web attachments = valence-3 nodes ----
val = np.zeros(len(nd), int)
for a, b in cells:
    val[a] += 1; val[b] += 1
junc = np.where(val >= 3)[0]
print("\n   valence-3 nodes (web/skin T-junctions): %s" % junc.tolist())
for j in junc:
    print("      node %2d at (y2,y3) = (%8.4f, %8.4f)" % (j, nd[j, 0], nd[j, 1]))

# ---- where is layup_6 relative to the junctions? ----
idx6 = [e for e, nm in enumerate(names) if nm == "layup_6"]
mid6 = np.array([0.5 * (nd[cells[e, 0]] + nd[cells[e, 1]]) for e in idx6])
print("\n   layup_6: %d elements, y2 %.4f..%.4f, y3 mean %.4f (spread %.4f)"
      % (len(idx6), mid6[:, 0].min(), mid6[:, 0].max(), mid6[:, 1].mean(), np.ptp(mid6[:, 1])))

upper = nd[junc][nd[junc][:, 1] > np.median(nd[:, 1])]
if len(upper):
    print("   junctions on the SAME (upper) surface as layup_6: y2 = %s"
          % np.array2string(np.sort(upper[:, 0]), precision=4))
    lo, hi = np.sort(upper[:, 0])[[0, -1]]
    inside = int(((mid6[:, 0] >= lo) & (mid6[:, 0] <= hi)).sum())
    print("   layup_6 elements lying BETWEEN the outermost upper junctions (y2 %.4f..%.4f): %d / %d"
          % (lo, hi, inside, len(idx6)))

# ---- which surface is in compression? ----
print("\n   sanity: flapwise M2<0 puts one surface in compression; layup_6 y3=%.3f vs section y3 range %.3f..%.3f"
      % (mid6[:, 1].mean(), nd[:, 1].min(), nd[:, 1].max()))
