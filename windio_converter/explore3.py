"""Resolve the IEA-22 blade cross-section at one span station r -> the concrete nd_arc segment layout,
layer stacks, webs, materials. This is the prototype data layer for the converter."""
import windIO, os
import numpy as np

p = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
d = windIO.load_yaml(p)
bl = d["components"]["blade"]
osb = bl["outer_shape"]; st = bl["structure"]
R = 0.5


def interp(spec, r):
    if isinstance(spec, dict) and "grid" in spec:
        return float(np.interp(r, spec["grid"], spec["values"]))
    if isinstance(spec, (int, float)):
        return float(spec)
    return None


# merge anchor lookup: top-level structure.anchors + each web's anchors
anch = {}
for a in st["anchors"]:
    anch[a["name"]] = a
for w in st["webs"]:
    for a in w.get("anchors", []):
        anch[a["name"]] = a


def resolve(spec, r):
    """spec = {grid,values} direct OR {anchor:{name,handle}} -> anchors[name][handle]."""
    if spec is None:
        return None
    if "anchor" in spec:
        ref = spec["anchor"]; a = anch[ref["name"]]
        return interp(a[ref["handle"]], r)
    return interp(spec, r)


print("=== station r=%.2f scalars ===" % R)
for k in ("chord", "twist", "rthick", "section_offset_y"):
    print("  %-16s = %.4f" % (k, interp(osb[k], R)))

print("\n=== anchor nd_arc positions at r=%.2f ===" % R)
brk = {}
for name, a in anch.items():
    s = resolve(a.get("start_nd_arc"), R) if "start_nd_arc" in a else None
    e = resolve(a.get("end_nd_arc"), R) if "end_nd_arc" in a else None
    print("  %-24s start=%s end=%s" % (name, s, e))
    if s is not None:
        brk[name + ".s"] = s
    if e is not None:
        brk[name + ".e"] = e

print("\n=== layers active at r=%.2f (start->end nd_arc, thickness, fiber) ===" % R)
for L in st["layers"]:
    s = resolve(L.get("start_nd_arc"), R); e = resolve(L.get("end_nd_arc"), R)
    th = interp(L.get("thickness"), R); fo = interp(L.get("fiber_orientation"), R)
    if th and th > 1e-6:
        print("  %-22s mat=%-18s s=%.4f e=%.4f t=%.4f fiber=%.1f"
              % (L["name"], L["material"], s if s is not None else -9, e if e is not None else -9, th, fo or 0))

print("\n=== materials (all) ===")
for m in d["materials"]:
    print("  %-20s orth=%s E=%s G=%s nu=%s rho=%s" % (m["name"], m.get("orth"), m.get("E"),
                                                      m.get("G"), m.get("nu"), m.get("rho")))

print("\n=== airfoil coordinate ordering (FFA-W3-270blend) ===")
for a in d["airfoils"]:
    if a["name"] == "FFA-W3-270blend":
        x = a["coordinates"]["x"]; y = a["coordinates"]["y"]
        print("  n=%d  first3 (x,y)=%s  mid=%s  last3=%s"
              % (len(x), list(zip(x[:3], y[:3])), (x[len(x)//2], y[len(y)//2]), list(zip(x[-3:], y[-3:]))))
        print("  x range %.3f..%.3f  (TE=1, LE=0 typically)" % (min(x), max(x)))
