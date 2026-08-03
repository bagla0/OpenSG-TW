'''
gen_shell51.py  --  51 uniform REAL cross-sections along the IEA-22 blade -> 1D shell YAML + PreVABS XML
================================================================================================
51 uniform span stations eta_i = i/50 (i=0..50), names iea_s00 .. iea_s50.  windIO is continuous,
so build_cross_section(blade, eta) gives the REAL section at any eta (no interpolation).  For each:

    cs = build_cross_section(blade, eta)
    emit_opensg_yaml(cs, shell51/1d_yaml/iea_sNN_shell.yaml)          # 1D shell SG YAML (RM input), fraction=0.5
    emit_prevabs   (cs, shell51/xml, name='iea_sNN', mesh_size=0.01, lam_reg=SHARED)   # XML (shared materials.xml)

ONE shared lam_reg dict across all 51 stations -> a single materials.xml (global lamina index) serves
every station (the pattern that fixed the tip lamina collision).  Shell-only: no .sg / no solid.
A degenerate tip (eta=1.0) is recorded and skipped, not aborted.
================================================================================================
'''
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # iea_all_stations


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
if IO not in sys.path:
    sys.path.insert(0, IO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # no jax / GPU needed here
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs  # noqa: E402

L_BLADE = 138.204                                   # m (for the span-position report only)

# ---- SINGLE reference-surface knob for the WHOLE pipeline (yaml -> RM homo -> BeamDyn -> dehom) ----
# "center" = laminate mid-surface (fraction 0.5, the default the RM/Kirchhoff shell assumes);
# "oml" = outer mould line (fraction 0.0).  It is written into every 1-D yaml's `reference` field and
# dehom_rm.build_rm_bundle reads it back, so changing THIS ONE line propagates everywhere automatically.
REFERENCE = "center"
FRACTION = {"center": 0.5, "oml": 0.0}[REFERENCE]


def main():
    windio = os.path.join(BASE, "IEA-22-280-RWT.yaml")
    y1d = os.path.join(HERE, "1d_yaml")
    xmld = os.path.join(HERE, "xml")
    for d in (y1d, xmld, os.path.join(HERE, "out")):
        os.makedirs(d, exist_ok=True)

    blade = load_blade(windio)
    print("windIO =", windio)
    print("out    =", HERE)
    print("51 uniform stations eta_i = i/50, i=0..50  (segment = L/50 = %.3f m)" % (L_BLADE / 50.0),
          flush=True)

    lam_reg = {}                                    # SHARED across all 51 -> one materials.xml
    ok_y = ok_x = 0
    fails = []
    for i in range(51):
        eta = i / 50.0
        name = "iea_s%02d" % i
        t0 = time.time()
        try:
            cs = build_cross_section(blade, r=eta, mesh_size=0.01)
            yp = os.path.join(y1d, name + "_shell.yaml")
            emit_opensg_yaml(cs, yp, fraction=FRACTION)             # records `reference` in the yaml
            gy = os.path.exists(yp)
            emit_prevabs(cs, xmld, name=name, mesh_size=0.01, lam_reg=lam_reg)
            gx = os.path.exists(os.path.join(xmld, name + ".xml"))
            ok_y += bool(gy)
            ok_x += bool(gx)
            print("[%-8s] eta=%.4f  z=%6.2fm  chord=%5.2f  webs=%d  yaml=%s  xml=%s  [%.1fs]"
                  % (name, eta, eta * L_BLADE, cs["chord"], len(cs["webs"]), gy, gx, time.time() - t0),
                  flush=True)
        except Exception as e:
            fails.append((name, eta, repr(e)[:200]))
            print("[%-8s] eta=%.4f  ERR %s" % (name, eta, repr(e)[:200]), flush=True)

    print("\n%d/51 shell YAML, %d/51 XML  (materials.xml laminae = %d)" % (ok_y, ok_x, len(lam_reg)),
          flush=True)
    if fails:
        print("FAILED stations:")
        for nm, eta, e in fails:
            print("  [%s] eta=%.4f  %s" % (nm, eta, e))


if __name__ == "__main__":
    main()
