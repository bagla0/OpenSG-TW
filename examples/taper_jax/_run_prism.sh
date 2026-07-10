#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
timeout 1200 $PY -u examples/taper_jax/3_prismatic_iea_hybrid_check.py > /tmp/prism.log 2>&1
echo "RC=$?"
grep -E 'prismatic segment:|hybrid batches|dof=|diagonal:|max \|err\||Error|Traceback' /tmp/prism.log | head -20
