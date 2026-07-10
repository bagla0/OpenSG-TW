#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
timeout 2400 $PY -u examples/taper_jax/4_hybrid_iea_taper_vs_shell.py > /tmp/iea4.log 2>&1
echo "RC=$?"
grep -E '=== |solid hybrid:|shell quad segment:|dof=|diagonal:|max \|err\||wrote|Error|Traceback' /tmp/iea4.log | head -30
