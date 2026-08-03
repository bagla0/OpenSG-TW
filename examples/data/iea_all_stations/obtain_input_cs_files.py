'''
================================================================================================
 obtain_input_cs_files.py  --  windIO  ->  PreVABS .xml   (STEP 1 of 2; xml only)
================================================================================================
Reads a windIO v2 blade, finds the airfoil-definition stations
(outer_shape.airfoils[].spanwise_position), and for each writes ONE PreVABS cross-section XML
(the common interface).  It is windIO -> xml, nothing else:

      windIO(r) ──build_cross_section──► contour + webs + layup ──emit_prevabs──► <name>.xml

ALL stations share ONE materials.xml: every windIO material card + a GLOBAL lamina registry, so
every station's layup resolves against the same file (a per-station lamina index would collide in
a shared xml/ folder -- that is exactly what broke the r=1 tip).

STEP 2 (separate, general): turn any .xml into the actual mesh files with

      python generate_mesh_file.py xml/            # .xml -> .sg + 2-D solid YAML + 1-D shell YAML

OUTPUT
  <out>/xml/   <name>.xml (one per station) + materials.xml (one, shared) + <name>.dat
               naming: <prefix>_r<round(r*1000)>

ARGUMENTS  (all optional)
  --windio PATH    windIO v2 blade YAML   (default: a windIO in this folder, else bundled IEA-22)
  --out DIR        output directory       (default: this script's directory)
  --r R            a SINGLE station r in [0,1]                 (overrides --stations)
  --stations LIST  comma list of r, e.g. 0.2,0.5,0.8          (default: ALL windIO airfoil stations)
  --mesh-size M    chord-normalised PreVABS mesh size written into the xml (default 0.01)
  --prefix P       output-file name prefix                    (default "iea")

EXAMPLES
  python obtain_input_cs_files.py                       # ALL airfoil stations -> xml/
  python obtain_input_cs_files.py --r 0.247            # one station
  python obtain_input_cs_files.py --stations 0.2,0.5,0.8
================================================================================================
'''
import argparse
import glob
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # no jax / no GPU needed (no homogenization)
from opensg_io import load_blade, build_cross_section, emit_prevabs   # noqa: E402


def airfoil_stations(windio_path):
    d = yaml.safe_load(open(windio_path))
    af = d["components"]["blade"]["outer_shape"]["airfoils"]
    return sorted(float(a["spanwise_position"]) for a in af)


def default_windio():
    for f in sorted(glob.glob(os.path.join(HERE, "*.yaml"))):
        try:
            if "windIO_version" in yaml.safe_load(open(f)):
                return f
        except Exception:
            pass
    return os.path.expanduser("~/OpenSG-TW-claude/examples/data/windio/IEA-22-280-RWT.yaml")


def main():
    ap = argparse.ArgumentParser(description="windIO -> per-station PreVABS .xml (step 1; xml only)")
    ap.add_argument("--windio", default=None, help="windIO v2 blade YAML (default: one in this dir, else IEA-22)")
    ap.add_argument("--out", default=HERE, help="output directory (default: this script's directory)")
    ap.add_argument("--r", type=float, default=None, help="a SINGLE station r (0..1); overrides --stations")
    ap.add_argument("--stations", default=None, help="comma list of r; default = all windIO airfoil stations")
    ap.add_argument("--mesh-size", type=float, default=0.01, help="chord-normalised PreVABS mesh size")
    ap.add_argument("--prefix", default="iea", help="output-file name prefix")
    a = ap.parse_args()

    windio = a.windio or default_windio()
    if a.r is not None:
        stations = [a.r]
    elif a.stations:
        stations = [float(x) for x in a.stations.split(",")]
    else:
        stations = airfoil_stations(windio)

    xmldir = os.path.join(a.out, "xml")
    os.makedirs(xmldir, exist_ok=True)
    blade = load_blade(windio)
    print("windIO   =", windio)
    print("out      =", xmldir)
    print("stations (%d) = %s" % (len(stations), ", ".join("%.4f" % s for s in stations)), flush=True)

    lam_reg = {}                        # shared across stations -> ONE complete materials.xml (global lamina index)
    ok = 0
    for r in stations:
        tag = "%s_r%04d" % (a.prefix, round(r * 1000))
        t0 = time.time()
        try:
            cs = build_cross_section(blade, r=r, mesh_size=a.mesh_size)
            emit_prevabs(cs, xmldir, name=tag, mesh_size=a.mesh_size, lam_reg=lam_reg)   # extends lam_reg + rewrites materials.xml
            good = os.path.exists(os.path.join(xmldir, tag + ".xml"))
            ok += bool(good)
            print("[%-10s] r=%.4f  chord=%5.2f  webs=%d  xml=%s  [%.1fs]"
                  % (tag, r, cs["chord"], len(cs["webs"]), good, time.time() - t0), flush=True)
        except Exception as e:
            print("[%-10s] r=%.4f  ERR %s" % (tag, r, repr(e)[:180]), flush=True)

    print("\n%d/%d stations -> xml + one materials.xml (%d laminae). Next: python generate_mesh_file.py %s"
          % (ok, len(stations), len(lam_reg), xmldir), flush=True)


if __name__ == "__main__":
    main()
