'''gen_s10_iml.py -- windIO-native IML 1-D yaml for station s10 (r/R=0.2) only:
emit_opensg_yaml(fraction=1.0) puts the contour ON the inner mold line (plies stacked
outward), then the same reference-axis shift as the center/OML sets.  For the r=0.2
three-reference dehomogenization comparison.  -> 1d_yaml_iml/iea_s10_shell.yaml
'''
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from refaxis_shift51 import shift_file


def find_io_root():
    cands = [os.path.expanduser("~/OpenSG-TW-claude/third_party/OpenSG_io"),
             os.path.expanduser("~/OpenSG_io")]
    d = HERE
    for _ in range(10):
        cands.append(os.path.join(d, "third_party", "OpenSG_io"))
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    for c in cands:
        if os.path.isdir(os.path.join(c, "opensg_io")):
            return c
    return None


sys.path.insert(0, find_io_root())
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml  # noqa: E402

out = os.path.join(HERE, "1d_yaml_iml")
os.makedirs(out, exist_ok=True)
blade = load_blade(os.path.join(BASE, "IEA-22-280-RWT.yaml"))
cs = build_cross_section(blade, r=0.2, mesh_size=0.01)
yp = os.path.join(out, "iea_s10_shell.yaml")
emit_opensg_yaml(cs, yp, fraction=1.0)
r = shift_file(yp)
print("wrote", yp, "shift:", r)
