#!/usr/bin/env bash
# Make the taper example self-contained: current opensg_io + the IEA windIO in data/.
set -e
TW=~/OpenSG-TW-claude
# 1. sync the bundled opensg_io to the live repo (which carries the ply-conforming fix)
rm -rf "$TW/third_party/OpenSG_io/opensg_io"
cp -r ~/OpenSG_io/opensg_io "$TW/third_party/OpenSG_io/opensg_io"
echo "synced third_party/OpenSG_io/opensg_io ($(ls $TW/third_party/OpenSG_io/opensg_io/*.py | wc -l) modules)"
# 2. windIO source mesh data into examples/data
mkdir -p "$TW/examples/data/windio"
cp ~/OpenSG_io/examples/data/IEA-22-280-RWT.yaml "$TW/examples/data/windio/"
echo "copied IEA-22-280-RWT.yaml -> examples/data/windio/"
ls -la "$TW/examples/data/windio/"
