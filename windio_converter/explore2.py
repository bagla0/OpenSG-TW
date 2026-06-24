"""Dump concrete IEA-22 structure values: anchors, layer spans+thickness, webs, airfoil coords."""
import windIO, os, json
import numpy as np

p = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
d = windIO.load_yaml(p)
bl = d["components"]["blade"]
osb = bl["outer_shape"]; st = bl["structure"]


def show(v, n=6):
    return json.dumps(v, default=str)[:400]


print("=== reference_axis ===")
for k in ("x", "y", "z"):
    ra = bl["reference_axis"][k]
    print("  %s: grid[0,-1]=%.3f..%.3f  values[0,-1]=%.3f..%.3f n=%d"
          % (k, ra["grid"][0], ra["grid"][-1], ra["values"][0], ra["values"][-1], len(ra["grid"])))

print("\n=== outer_shape scalar distributions (grid, values) ===")
for k in ("chord", "twist", "rthick", "section_offset_y"):
    v = osb[k]
    print("  %-16s n=%d grid %.2f..%.2f val %.4g..%.4g" % (k, len(v["grid"]), v["grid"][0], v["grid"][-1],
                                                           v["values"][0], v["values"][-1]))

print("\n=== outer_shape.airfoils (grid -> airfoil name) ===")
for a in osb["airfoils"]:
    print("  ", show(a))

print("\n=== anchors (%d) ===" % len(st["anchors"]))
for a in st["anchors"]:
    print("  ", a.get("name"), "->", show({k: v for k, v in a.items() if k != "name"}))

print("\n=== webs (%d) ===" % len(st["webs"]))
for w in st["webs"]:
    print("  ", w.get("name"), "->", show({k: v for k, v in w.items() if k != "name"}))

print("\n=== layers (%d) -- name, material, start/end nd_arc, thickness ===" % len(st["layers"]))
for L in st["layers"]:
    sa, ea = L.get("start_nd_arc"), L.get("end_nd_arc")
    th = L.get("thickness")
    thsig = ("grid n=%d val %.4g..%.4g" % (len(th["grid"]), min(th["values"]), max(th["values"]))) if isinstance(th, dict) else str(th)
    print("  %-22s mat=%-16s fo=%s\n       start=%s end=%s\n       thick=%s"
          % (L.get("name"), L.get("material"), show(L.get("fiber_orientation")), show(sa), show(ea), thsig))

print("\n=== airfoils (geometry) ===")
for a in d["airfoils"]:
    c = a.get("coordinates", {})
    xy = c if isinstance(c, dict) else {}
    nx = len(xy.get("x", [])) if isinstance(xy, dict) else 0
    print("  %-22s rthick=%s  coords n=%d keys=%s" % (a.get("name"), a.get("rthick"), nx,
                                                      list(xy.keys()) if isinstance(xy, dict) else type(c)))
