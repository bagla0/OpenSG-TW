#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
find ~/OpenSG_io ~/OpenSG-TW-claude/third_party/OpenSG_io -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
$PY -c "
import sys, os, inspect
sys.path.insert(0, os.path.expanduser('~/OpenSG-TW-claude/examples/taper'))
sys.path.insert(0, os.path.expanduser('~/OpenSG_io'))
import taper_common
import opensg_io.hex_loft as HL
print('hex_loft loaded from:', HL.__file__)
print('min_sep patch active:', '_min_sep' in inspect.getsource(HL.build_section_mesh))
"
timeout 1200 $PY -u examples/taper_jax/_scan_02h.py 2>&1 | grep -E 'r=|Error' | head -8
