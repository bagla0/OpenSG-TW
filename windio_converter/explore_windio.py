"""Explore the IEA-22-280-RWT windIO structure -- map what the converter must consume."""
import windIO
import os
import numpy as np

p = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
d = windIO.load_yaml(p)


def kt(v):
    if isinstance(v, dict):
        return "dict{%s}" % ", ".join(list(v.keys())[:12])
    if isinstance(v, list):
        return "list[%d]" % len(v)
    return type(v).__name__


print("TOP:", kt(d))
print("components:", kt(d.get("components")))
bl = d["components"]["blade"]
print("\nblade keys:", list(bl.keys()))
for k in bl:
    print("  blade.%s -> %s" % (k, kt(bl[k])))

# outer shape
os_key = "outer_shape_bem" if "outer_shape_bem" in bl else "outer_shape"
osb = bl.get(os_key, {})
print("\n%s keys:" % os_key, list(osb.keys()))
for k in osb:
    print("   .%s -> %s" % (k, kt(osb[k])))

# internal structure
isk = "internal_structure_2d_fem" if "internal_structure_2d_fem" in bl else "structure"
ist = bl.get(isk, {})
print("\n%s keys:" % isk, list(ist.keys()))
for k in ist:
    print("   .%s -> %s" % (k, kt(ist[k])))

# webs
webs = ist.get("webs", [])
print("\nWEBS:", len(webs))
for w in webs[:3]:
    print("  web:", {k: kt(v) for k, v in w.items()})

# layers
layers = ist.get("layers", [])
print("\nLAYERS:", len(layers))
for L in layers[:6]:
    print("  layer '%s': keys=%s" % (L.get("name"), list(L.keys())))

# airfoils + materials
af = d.get("airfoils", bl.get("airfoils"))
print("\nAIRFOILS:", kt(af))
if isinstance(af, list) and af:
    print("  first airfoil keys:", list(af[0].keys()))
mats = d.get("materials")
print("MATERIALS:", kt(mats))
if isinstance(mats, list) and mats:
    print("  example material keys:", list(mats[0].keys()))
    for m in mats[:8]:
        print("    -", m.get("name"), "orth=", m.get("orth"), "E=", m.get("E"))
