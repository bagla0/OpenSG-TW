#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=~/miniconda3/envs/opensg_2_0/bin/python
echo "=== regenerate circle (tube) meshes ==="
(cd mitc_rm_segment && timeout 600 $PY -u taper_study.py gen 2>&1 | tail -3)
ls mitc_rm_segment/out/taper_study/meshes/solid_*aR070* 2>/dev/null
for M in mitc_rm_segment/out/taper_study/meshes/solid_thick_m45_aR070.yaml \
         mitc_rm_segment/out/taper_study/meshes/solid_thin_m45_aR070.yaml; do
  [ -f "$M" ] || continue
  echo; echo "######## TUBE $(basename $M) ########"
  timeout 900 $PY -u examples/taper_jax/1_run_solid_taper_jax.py "$M" --fenics 2>&1 \
    | grep -E 'max \|err\||elements=|SEGMENT Timoshenko|diagonal: EA' | head -8
done
