#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate opensg_2_0
cd ~/OpenSG-TW-claude/third_party/OpenSG_io
WIN=~/OpenSG-TW-claude/examples/data/windio/IEA-22-280-RWT.yaml
rm -rf ~/claude_tmp/span_out; mkdir -p ~/claude_tmp/span_out
# IEA airfoil stations in [0.2, 1.0): the file's geometry stations (no interpolation)
for r in 0.2470 0.3993 0.5336 0.7389 0.9800; do
  nm=iea_r$(python -c "print('%04d'%round($r*1000))")
  python scripts/convert_station.py --yaml $WIN --r $r --mesh-size 0.01 \
     --out ~/claude_tmp/span_out/$nm --name $nm 2>&1 | grep -E "station r=|PreVABS"
done
echo "==== XMLs produced ===="
find ~/claude_tmp/span_out -name "*.xml" | sort
