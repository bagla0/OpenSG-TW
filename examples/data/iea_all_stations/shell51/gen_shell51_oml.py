'''gen_shell51_oml.py -- SINGLE-RUN regeneration of ALL 51 station 1-D shell yamls at the
OML reference, directly from windIO, into the SEPARATE folder 1d_yaml_oml51/.

Identical chain to gen_shell51.py (build_cross_section at eta_i=i/50 -> emit_opensg_yaml)
with the single reference knob set to "oml" (fraction=0.0: contour ON the outer mold
line, plies stacked inward, `reference: oml` recorded in every yaml so build_rm_bundle
follows automatically), followed by the same reference-axis shift as the center set
(refaxis_shift51.shift_file: x2 -= section_offset_y(eta)).  The contour is windIO-native
-- NOT the node-by-node outward offset of the mid-surface contour -- so it carries none
of that construction's node-scale noise.

    ~/miniconda3/envs/opensg_2_0/bin/python gen_shell51_oml.py
'''
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # iea_all_stations
sys.path.insert(0, HERE)
from refaxis_shift51 import shift_file             # same shift + sidecar marker as the center set


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


IO = find_io_root()
if IO is None:
    sys.exit("Could not locate the OpenSG_io package (opensg_io/).")
sys.path.insert(0, IO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml  # noqa: E402

L_BLADE = 138.204
REFERENCE = "oml"                                  # THE single reference knob
FRACTION = {"center": 0.5, "oml": 0.0}[REFERENCE]


def main():
    windio = os.path.join(BASE, "IEA-22-280-RWT.yaml")
    y1d = os.path.join(HERE, "1d_yaml_oml51")
    os.makedirs(y1d, exist_ok=True)
    blade = load_blade(windio)
    print("windIO =", windio)
    print("out    =", y1d, " (reference=%s, fraction=%.1f)" % (REFERENCE, FRACTION), flush=True)

    ok = 0
    fails = []
    for i in range(51):
        eta = i / 50.0
        name = "iea_s%02d" % i
        t0 = time.time()
        try:
            cs = build_cross_section(blade, r=eta, mesh_size=0.01)
            yp = os.path.join(y1d, name + "_shell.yaml")
            emit_opensg_yaml(cs, yp, fraction=FRACTION)
            r = shift_file(yp)                       # LE -> reference-axis origin
            ok += 1
            print("[%-8s] eta=%.4f  z=%6.2fm  chord=%5.2f  webs=%d  shift=%s  [%.1fs]"
                  % (name, eta, eta * L_BLADE, cs["chord"], len(cs["webs"]),
                     "%.3f" % r[1] if r else "-", time.time() - t0), flush=True)
        except Exception as e:
            fails.append((name, eta, repr(e)[:160]))
            print("[%-8s] eta=%.4f  ERR %s" % (name, eta, repr(e)[:160]), flush=True)

    print("\n%d/51 OML shell yamls -> %s" % (ok, y1d), flush=True)
    if fails:
        print("FAILED stations:")
        for nm, eta, e in fails:
            print("  [%s] eta=%.4f  %s" % (nm, eta, e))


if __name__ == "__main__":
    main()
