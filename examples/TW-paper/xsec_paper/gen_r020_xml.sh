#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate opensg_2_0
cd ~/OpenSG-TW-claude/third_party/OpenSG_io
rm -rf ~/claude_tmp/r020_out
python scripts/convert_station.py \
  --yaml ~/OpenSG-TW-claude/examples/data/windio/IEA-22-280-RWT.yaml \
  --r 0.2 --mesh-size 0.01 --out ~/claude_tmp/r020_out --name iea_r020
echo "==== produced ===="
find ~/claude_tmp/r020_out -type f | sort
echo "==== xml head ===="
head -25 ~/claude_tmp/r020_out/iea_r020_prevabs/*.xml 2>/dev/null
