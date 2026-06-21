"""Compare the WEB material frame (e1 fibre, e2 tangent, e3 ply-normal) between the 1D-shell YAML
and the 2D-solid YAML, at the two web stations x=-0.169 and x=+0.496 (model coords).
The solid is the reference; the shell web e3 must match it (consistent ply-stacking direction)."""
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SHELL = os.path.join(HERE, "shell_ref_f020_connect.yaml")
SOLID = os.path.join(HERE, "..", "yaml_solid", "solid_f020.yaml")
WX = [-0.1691, 0.4959]


def load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([[float(v) for v in str(r[0]).split()] for r in d["nodes"]])
    elems = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    oris = np.array([[float(v) for v in r] for r in d["elementOrientations"]])
    cent = np.array([nodes[e].mean(axis=0) for e in elems])
    return nodes, elems, oris, cent, d


# ---- 1D shell web elements (layup_4) ----
ns, es, osh, cs, ds = load(SHELL)
webset = set(ds["sets"]["element"][4]["labels"])
print("=== 1D SHELL web elements (layup_4) ===")
for lab in sorted(webset):
    o = osh[lab - 1]
    e1, e2, e3 = o[0:3], o[3:6], o[6:9]
    print("  elem %d @x=%.3f:  e1=[%+.2f %+.2f %+.2f] e2=[%+.2f %+.2f %+.2f] e3=[%+.2f %+.2f %+.2f]"
          % (lab, cs[lab - 1][0], *e1, *e2, *e3))

# ---- 2D solid web-region elements (vertical strip at each WX) ----
nsd, esd, osd, csd, dd = load(SOLID)
print("\n=== 2D SOLID web-region elements (vertical strip near each web x) ===")
for wx in WX:
    sel = np.where(np.abs(csd[:, 0] - wx) < 0.004)[0]
    sel = sel[np.argsort(csd[sel, 1])]
    print("  web x~%.3f : %d solid elements" % (wx, len(sel)))
    for k in sel[::max(1, len(sel) // 6)]:
        o = osd[k]; e1, e2, e3 = o[0:3], o[3:6], o[6:9]
        print("    @y=%+.3f  e1=[%+.2f %+.2f %+.2f] e2=[%+.2f %+.2f %+.2f] e3=[%+.2f %+.2f %+.2f]"
              % (csd[k][1], *e1, *e2, *e3))
