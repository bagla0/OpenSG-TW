"""Convert one windIO blade station to OpenSG inputs: 1D-shell SG YAML + PreVABS XML (+ optional .sg).

Examples:
  python scripts/convert_station.py --r 0.5 --mesh-size 0.01 --out out/
  python scripts/convert_station.py --yaml path/to/blade.yaml --r 0.7 --prevabs /path/to/prevabs.exe --run
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import windIO
from opensg_io import WindIOBlade, build_cross_section, emit_opensg_yaml, emit_prevabs


def default_iea22():
    return os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")


def main():
    ap = argparse.ArgumentParser(description="windIO v2 station -> OpenSG 1D YAML + PreVABS XML")
    ap.add_argument("--yaml", default=default_iea22(), help="windIO v2 blade YAML (default: bundled IEA-22-280-RWT)")
    ap.add_argument("--r", type=float, required=True, help="non-dimensional span station, 0..1")
    ap.add_argument("--mesh-size", type=float, default=0.01, help="chord-normalised contour element size")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--name", default=None, help="basename (default iea22_rNNN)")
    ap.add_argument("--prevabs", default=os.environ.get("PREVABS_EXE"), help="path to prevabs(.exe) to also mesh the solid")
    ap.add_argument("--run", action="store_true", help="run prevabs.exe to produce the .sg (needs --prevabs/PREVABS_EXE)")
    a = ap.parse_args()

    name = a.name or ("xsec_r%03d" % round(a.r * 100))
    os.makedirs(a.out, exist_ok=True)
    blade = WindIOBlade(a.yaml)
    cs = build_cross_section(blade, a.r, mesh_size=a.mesh_size)
    shell = os.path.join(a.out, name + "_shell.yaml")
    emit_opensg_yaml(cs, shell)
    pvdir = os.path.join(a.out, name + "_prevabs")
    info = emit_prevabs(cs, pvdir, name=name, mesh_size=a.mesh_size)
    print("station r=%.3f chord=%.3f m twist=%.2f deg | %d laminates, %d webs"
          % (cs["r"], cs["chord"], cs["twist"], len(cs["laminates"]), len(cs["webs"])))
    print("  1D-shell SG :", shell)
    print("  PreVABS     :", os.path.join(pvdir, info["xml"]))

    if a.run:
        if not a.prevabs or not os.path.exists(a.prevabs):
            sys.exit("--run needs a valid --prevabs path or PREVABS_EXE env var")
        subprocess.run([a.prevabs, "-i", info["xml"], "--vabs", "--hm"], cwd=pvdir, check=True)
        sg = os.path.join(pvdir, name + ".sg")
        print("  2D-solid .sg:", sg, "OK" if os.path.exists(sg) else "FAILED")


if __name__ == "__main__":
    main()
