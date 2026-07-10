#!/usr/bin/env bash
cd ~/OpenSG-TW-claude/mitc_rm_segment/out
echo "=== dirs ==="; ls -d */ 2>/dev/null
echo "=== solid meshes ==="; ls taper_square/meshes/solid_* taper_study/meshes/solid_* 2>/dev/null | head -20
echo "=== results (refs) ==="; ls taper_square/results/ taper_study/results/ 2>/dev/null | head -40
echo "=== one mesh head ==="
M=$(ls taper_study/meshes/solid_*m45*.yaml taper_square/meshes/solid_*.yaml 2>/dev/null | head -1)
echo "mesh: $M"
head -6 "$M"
python3 - "$M" <<'EOF'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
print("keys:", list(d.keys()))
print("n nodes:", len(d["nodes"]), " node0:", d["nodes"][0])
print("n elems:", len(d["elements"]), " elem0:", d["elements"][0])
print("mat0:", {k: v for k, v in d["materials"][0].items() if k != "labels"})
print("sets:", [s["name"] for s in d.get("sets", {}).get("element", [])])
eo = d.get("elementOrientations")
print("orient0:", eo[0] if eo else None)
EOF
