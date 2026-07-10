#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=~/miniconda3/envs/opensg_2_0/bin/python
echo "=== available taper_study meshes ==="
ls mitc_rm_segment/out/taper_study/meshes/solid_* 2>/dev/null | head -12
for M in mitc_rm_segment/out/taper_square/meshes/solid_thick_m45_aR070.yaml \
         $(ls mitc_rm_segment/out/taper_study/meshes/solid_*m45*.yaml 2>/dev/null | head -1) \
         $(ls mitc_rm_segment/out/taper_study/meshes/solid_*.yaml 2>/dev/null | grep -v m45 | head -1); do
  [ -f "$M" ] || continue
  echo; echo "######## $M ########"
  timeout 600 $PY -u examples/taper_jax/1_run_solid_taper_jax.py "$M" --fenics 2>&1 \
    | grep -E 'max \|err\||elements=|diagonal: EA' | head -8
done
