#!/bin/bash
set -e
TD="/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
OUTDIR="/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/outputs"
mkdir -p "$OUTDIR"
export PYTHONPATH="$TD:$PYTHONPATH"
cd /tmp
echo "PYTHONPATH=$PYTHONPATH"
python3 /mnt/c/Users/bagla0/benchmark_oml_fenics.py "$OUTDIR/oml_fenics.txt"
