#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
find ~/OpenSG-TW-claude -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
timeout 1800 $PY -u examples/TW-paper/xsec_paper/ellipse/ell4cell.py conv 2>&1 | grep -E 'nc=|iso|m45|Error|Traceback' | tail -12
