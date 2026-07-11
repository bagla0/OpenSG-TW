#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
find ~/OpenSG-TW-claude ~/OpenSG_io -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
echo "############## CONVERGENCE (single tube) ##############"
timeout 900 $PY -u examples/TW-paper/xsec_paper/conv_single_tube.py 2>&1 | tail -20
echo; echo "############## THICKNESS SWEEP (IEA r=0.2) ##############"
timeout 1800 $PY -u examples/TW-paper/xsec_paper/thick_sweep_r020.py 2>&1 | tail -20
