"""diff_yaml.py -- the repo tests/data copy and the GitHub Shell_1DSG download of 1Dshell_15.yaml have
DIFFERENT md5s (20676 vs 21212 bytes).  The st15 dehom that validated to 0.35% vs VABS used the REPO copy;
the e3-sign audit that found ~23% flipped normals used the DOWNLOAD.  Before building the FSM on either, find
out whether the difference is cosmetic (formatting/precision) or substantive (geometry, layup, orientation).
"""
import os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
A = os.path.join(ROOT, "tests", "data", "1Dshell_15.yaml")
B = os.path.join(HERE, "Shell_1DSG", "1Dshell_15.yaml")


def _row(x):
    """Flatten a yaml row to floats. The two vintages differ: the download writes a real list of numbers,
    the repo copy writes a 1-element list holding a whitespace-joined string."""
    out = []
    for v in ([x] if isinstance(x, str) else x):
        if isinstance(v, str):
            out.extend(float(t) for t in v.split())
        else:
            out.append(float(v))
    return out


def load(p):
    d = yaml.safe_load(open(p))
    return d


da, db = load(A), load(B)
print("key sets equal:", set(da.keys()) == set(db.keys()))
print("  repo keys    :", sorted(da.keys()))
print("  download keys:", sorted(db.keys()))

na = np.array([_row(n)[:2] for n in da["nodes"]])
nb = np.array([_row(n)[:2] for n in db["nodes"]])
print("\nnodes: repo %s  download %s" % (na.shape, nb.shape))
if na.shape == nb.shape:
    d = np.linalg.norm(na - nb, axis=1)
    print("   max |dP| = %.3e m   median = %.3e m" % (d.max(), np.median(d)))
    if d.max() > 1e-9:
        w = np.argsort(-d)[:5]
        for i in w:
            print("     node %3d  repo=(%.8f, %.8f)  dl=(%.8f, %.8f)  d=%.3e"
                  % (i, na[i, 0], na[i, 1], nb[i, 0], nb[i, 1], d[i]))

ea = np.array([[int(x) for x in _row(e)] for e in da["elements"]])
eb = np.array([[int(x) for x in _row(e)] for e in db["elements"]])
print("\nelements: repo %s  download %s   identical=%s"
      % (ea.shape, eb.shape, ea.shape == eb.shape and bool((ea == eb).all())))

for k in ("sets", "sections", "materials", "elementOrientations"):
    xa, xb = da.get(k), db.get(k)
    same = (xa == xb)
    print("\n%-20s present repo=%s download=%s  identical=%s"
          % (k, xa is not None, xb is not None, same))
    if not same and xa is not None and xb is not None:
        if k == "sections":
            la = {s["elementSet"]: s["layup"] for s in xa}
            lb = {s["elementSet"]: s["layup"] for s in xb}
            print("   elementSets repo=%d download=%d  same names=%s"
                  % (len(la), len(lb), set(la) == set(lb)))
            for nm in sorted(set(la) & set(lb)):
                if la[nm] != lb[nm]:
                    print("   layup differs for %s:\n      repo=%s\n      dl  =%s" % (nm, la[nm], lb[nm]))
        elif k == "materials":
            ma = {m["name"]: m for m in xa}
            mb = {m["name"]: m for m in xb}
            print("   repo mats=%s" % sorted(ma))
            print("   dl   mats=%s" % sorted(mb))
            for nm in sorted(set(ma) & set(mb)):
                if ma[nm] != mb[nm]:
                    print("   material differs: %s\n      repo=%s\n      dl  =%s" % (nm, ma[nm], mb[nm]))
        elif k == "sets":
            sa = {g["name"]: list(g["labels"]) for g in xa["element"]}
            sb = {g["name"]: list(g["labels"]) for g in xb["element"]}
            print("   set names same=%s" % (set(sa) == set(sb)))
            for nm in sorted(set(sa) & set(sb)):
                if sa[nm] != sb[nm]:
                    print("   labels differ for %s: repo n=%d dl n=%d" % (nm, len(sa[nm]), len(sb[nm])))
        elif k == "elementOrientations":
            oa = np.array([_row(o) for o in xa]); ob = np.array([_row(o) for o in xb])
            print("   shapes repo=%s dl=%s" % (oa.shape, ob.shape))
            if oa.shape == ob.shape:
                dd = np.abs(oa - ob).max(axis=1)
                print("   max |dOrient| = %.3e   n elements differing >1e-9 : %d"
                      % (dd.max(), int((dd > 1e-9).sum())))
