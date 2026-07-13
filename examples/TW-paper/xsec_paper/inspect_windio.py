"""inspect_windio.py -- dump the ACTUAL spanwise grids and ply-thickness schedule of the windIO
blade (how the input is really given), and compare with the discrete OpenSG shell YAML at a station.
Usage: python inspect_windio.py
"""
import yaml, os, numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
d = yaml.safe_load(open(os.path.join(R, "examples/data/windio/IEA-22-280-RWT.yaml")))
bl = d["components"]["blade"]

print("windIO_version:", d.get("windIO_version"))
ra = bl["reference_axis"]
print("\n=== reference_axis (blade geometry backbone) ===")
for k in ["x", "y", "z"]:
    g = ra[k]["grid"]; v = ra[k]["values"]
    print("  %s: %3d grid pts  grid=[%.3f..%.3f]  value[-1]=%.3f" % (k, len(g), g[0], g[-1], v[-1]))

osh = bl["outer_shape"]
print("\n=== outer_shape (aerodynamic) -- each field has its OWN spanwise grid ===")
for k in ["chord", "twist", "rthick"]:
    if k in osh:
        g = osh[k]["grid"]
        print("  %-8s: %3d grid pts  [%.4f .. %.4f]" % (k, len(g), g[0], g[-1]))
af = osh.get("airfoils", [])
print("  airfoils : %d shapes at spanwise_position = %s" % (len(af), [round(a["spanwise_position"], 4) for a in af]))

st = bl["structure"]
lay = st["layers"]
print("\n=== structure.layers -- %d layers (plies/regions), each with its OWN thickness grid ===" % len(lay))
print("   %-20s %-10s ngrid  thickness schedule  (r, t_mm) -- linear between points = ply drop" % ("layer name", "material"))
for L in lay:
    th = L.get("thickness", {})
    g = th.get("grid", []); v = th.get("values", [])
    pts = [(round(float(gg), 3), round(float(vv) * 1e3, 2)) for gg, vv in zip(g, v)]
    # show only the non-zero span of the ply (where it actually exists) + a couple of zeros around it
    nzero = [i for i, (_, t) in enumerate(pts) if t > 1e-9]
    show = pts
    if nzero:
        a = max(0, nzero[0] - 1); b = min(len(pts), nzero[-1] + 2); show = pts[a:b]
    print("   %-20s %-10s %4d   %s" % (L["name"][:20], str(L.get("material", "?"))[:10], len(g), show[:9]))

print("\n=== structure.webs ===")
for w in st["webs"]:
    keys = [k for k in w.keys() if k != "name"]
    print("  %-8s keys=%s" % (w.get("name"), keys))

# --- compare with the discrete OpenSG shell YAML at r=0.2 ---
sh = yaml.safe_load(open(os.path.join(R, "examples/TW-paper/iea22_blade/data/shell_r020.yaml")))
print("\n=== OpenSG shell_r020.yaml (ONE station, layup baked to discrete element sets) ===")
print("  element sets:", [(g["name"], len(g["labels"])) for g in sh["sets"]["element"]])
for s in sh["sections"][:8]:
    lu = [(p[0] if isinstance(p, list) else p, round(float(p[1]) * 1e3, 2) if isinstance(p, list) else None) for p in s["layup"]]
    print("  %-10s layup(mat, t_mm) = %s" % (s["elementSet"], lu))
