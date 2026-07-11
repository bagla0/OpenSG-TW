#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
find ~/OpenSG_io ~/OpenSG-TW-claude/third_party/OpenSG_io -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
echo "=== 0.2h pairs (expect CLEAN) ==="
timeout 1200 $PY -u examples/taper_jax/_scan_02h.py 2>&1 | grep -E 'r=' | head -8
echo "=== FULL-thickness adjacent pairs (old folds: 25-124 inverted) ==="
timeout 1500 $PY - <<'EOF'
import sys, os
sys.path.insert(0, os.path.expanduser('~/OpenSG-TW-claude'))
sys.path.insert(0, os.path.expanduser('~/OpenSG-TW-claude/examples/taper'))
sys.path.insert(0, os.path.expanduser('~/OpenSG_io'))
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section
from opensg_io.hex_loft import hex_between_sections
from opensg_io.conformity import min_scaled_jacobian
blade = load_blade(WINDIO)
for r1, r2 in [(0.0487, 0.0665), (0.2470, 0.3993), (0.3993, 0.5336), (0.7389, 0.9800)]:
    cs1 = build_cross_section(blade, r1, mesh_size=0.02)
    cs2 = build_cross_section(blade, r2, mesh_size=0.02)
    res = hex_between_sections(cs1, cs2, blade_span_z(blade, r1), blade_span_z(blade, r2),
                               nr=4, nsp=8, nw=3, mesh_size=0.02)
    msj, ninv = min_scaled_jacobian(res["nodes"], res["hexes"])
    print("FULL r=%.4f->%.4f : inverted=%-4d minSJ=%+.3f %s"
          % (r1, r2, ninv, msj, "CLEAN" if ninv == 0 else ""), flush=True)
EOF
