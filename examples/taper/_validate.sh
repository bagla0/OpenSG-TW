#!/usr/bin/env bash
set -e
cd ~/OpenSG-TW-claude/examples/taper
PY=~/miniconda3/envs/opensg_2_0/bin/python
export PYTHONIOENCODING=utf-8
OUT=out_r020_030
echo "########## 1: solid mesh"; $PY 1_generate_solid_mesh.py 0.20 0.30 $OUT
echo "########## 2: shell mesh"; $PY 2_generate_shell_mesh.py 0.20 0.30 $OUT
echo "########## 3: solid boundary L"; $PY 3_get_beam_props_from_solid_boundary.py $OUT/solid_boundary_L.yaml
echo "########## 5: shell boundary L"; $PY 5_get_beam_props_from_shell_boundary.py $OUT/shell_boundary_L.yaml
echo "########## 6: shell segment";    $PY 6_get_beam_props_from_shell_segment.py $OUT/shell_segment.yaml
echo "########## 4: solid segment";    $PY 4_get_beam_props_from_solid_segment.py $OUT/solid_segment.yaml
