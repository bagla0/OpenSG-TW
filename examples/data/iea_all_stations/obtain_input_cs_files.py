'''
================================================================================================
 obtain_input_cs_files.py  --  README
 windIO  ->  cross-section INPUT files   (general: any windIO v2 blade, any/all stations, any type)
================================================================================================

WHAT IT DOES
  Reads a windIO v2 blade, finds the airfoil-definition stations
  (outer_shape.airfoils[].spanwise_position), and for each requested station builds the
  cross-section INPUT files.  The PreVABS XML is the COMMON PATHWAY -- windIO -> XML -> everything:

      windIO(r)  --build_cross_section-->  contour + webs + layup
          |                                       |
          |-- emit_opensg_yaml --------------> 1-D shell SG YAML                 (RM / KL shell)
          |
          |-- emit_prevabs -> <name>.xml  --(prevabs -i xml --vabs --hm)-->  <name>.sg  (VABS input)
                                                              |
                                                              |-- convert_sg_to_yaml --> 2-D solid YAML

  This step is FAST -- PreVABS meshing only, NO homogenization.  VABS is NOT executed here:
  "--hm" is only the analysis-mode flag; without -e/--execute PreVABS merely WRITES the .sg.

OUTPUT  (one folder per type, under --out)
  <out>/xml/       <name>.xml + materials.xml + <name>.dat    PreVABS common-pathway inputs
  <out>/sg/        <name>.sg                                  VABS input mesh   (run VABS -> .K)
  <out>/2d_yaml/   <name>_solid.yaml                          2-D solid OpenSG YAML (JAX / FEniCS homo)
  <out>/1d_yaml/   <name>_shell.yaml                          1-D shell SG YAML     (RM homo)
  naming: <prefix>_r<round(r*1000)>   e.g.  iea_r0247  = r 0.247

ARGUMENTS  (all optional)
  --windio PATH    windIO v2 blade YAML    (default: a windIO in this folder, else bundled IEA-22-280)
  --out DIR        output directory        (default: this script's own directory)
  --r R            a SINGLE station r in [0,1]         (overrides --stations)
  --stations LIST  comma list of r, e.g. 0.2,0.5,0.8   (default: ALL windIO airfoil stations)
  --types LIST     comma subset of {sg,1d,2d} to build (default: all).  2d implies sg;
                   "--types 1d" skips PreVABS entirely (near-instant).
  --mesh-size M    chord-normalised PreVABS mesh size  (default 0.01)
  --prefix P       output-file name prefix             (default "iea")
  --prevabs PATH   prevabs binary                      (default: auto-detect / $PREVABS_EXE)

EXAMPLES
  python obtain_input_cs_files.py                       # ALL airfoil stations, ALL types (sg,1d,2d)
  python obtain_input_cs_files.py --r 0.5336            # one station, all types
  python obtain_input_cs_files.py --r 0.5 --types 2d    # one station, only the 2-D solid YAML
  python obtain_input_cs_files.py --types 1d            # all stations, only the 1-D shell YAML
  python obtain_input_cs_files.py --stations 0.2,0.5,0.8 --types sg,2d
  python obtain_input_cs_files.py --windio other_blade.yaml --out other_out/

NEXT STEPS -- homogenization (run separately, step by step)
  VABS  .K    :  cd <out>/sg   &&   VABS.exe <name>.sg          -> Timoshenko 6x6 stiffness (.K)
  JAX/FEniCS  :  compute_timo_from_yaml(<out>/2d_yaml/<name>_solid.yaml)   -> 2-D solid 6x6
  RM   6x6    :  ring_6dof(load_ring(<out>/1d_yaml/<name>_shell.yaml))     -> RM shell 6x6

Every station is wrapped in try/except: a failure is recorded and the run continues.
================================================================================================
'''
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def find_io_root():
    """Locate the OpenSG_io package root (the dir containing opensg_io/ and scripts/), so this
    script works whether it lives in OpenSG_io/scripts/ or copied into a data directory."""
    cands = [os.path.abspath(os.path.join(HERE, "..")),
             os.path.expanduser("~/OpenSG-TW-claude/third_party/OpenSG_io"),
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
    sys.exit("Could not locate the OpenSG_io package (opensg_io/). Run from the repo, or set it on PATH.")
for q in (IO, os.path.expanduser("~/OpenSG_io")):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # no GPU / no jax needed (no homogenization)
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs

CONVERT = os.path.join(IO, "scripts", "convert_sg_to_yaml.py")   # .sg -> 2-D solid OpenSG YAML


def find_prevabs(user):
    for cand in (user, os.environ.get("PREVABS_EXE")):
        if cand and os.path.exists(cand):
            return cand
    for pat in ("~/OpenSG_io/third_party/prevabs_bin/**/prevabs",
                "~/OpenSG_io/third_party/prevabs_bin/**/prevabs.exe"):
        g = sorted(glob.glob(os.path.expanduser(pat), recursive=True))
        if g:
            return g[-1]
    return None


def airfoil_stations(windio_path):
    d = yaml.safe_load(open(windio_path))
    af = d["components"]["blade"]["outer_shape"]["airfoils"]
    return sorted(float(a["spanwise_position"]) for a in af)


def default_windio():
    """A windIO in the script's own directory (e.g. copied into iea_all_stations/) wins, else the
    bundled IEA-22 blade."""
    for f in sorted(glob.glob(os.path.join(HERE, "*.yaml"))):
        try:
            if "windIO_version" in yaml.safe_load(open(f)):
                return f
        except Exception:
            pass
    return os.path.expanduser("~/OpenSG-TW-claude/examples/data/windio/IEA-22-280-RWT.yaml")


def sh(cmd, cwd=None, timeout=1200):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return p.returncode == 0, (p.stdout or "")


G = {}          # shared context for worker processes (inherited via fork on Linux)


def process_one(r):
    """Build the input files for ONE station r.  Uses the module-global G set up in main() (the
    windIO blade, output dirs, prevabs path, type flags) -- inherited by fork so nothing is pickled
    except the float r and the small result dict.  Safe to run in a process Pool."""
    a = G["a"]
    tag = "%s_r%04d" % (a.prefix, round(r * 1000))
    rec = dict(r=r, tag=tag, chord=float("nan"), nweb=0, sg=False, y2d=False, y1d=False, err="")
    t0 = time.time()
    try:
        cs = build_cross_section(G["blade"], r=r, mesh_size=a.mesh_size)
        rec["chord"] = float(cs["chord"]); rec["nweb"] = len(cs["webs"])
        if G["want_1d"]:
            shell = os.path.join(G["dirs"]["1d_yaml"], tag + "_shell.yaml")
            emit_opensg_yaml(cs, shell); rec["y1d"] = os.path.exists(shell)
        if G["want_sg"]:
            pv = os.path.join(G["work"], tag); os.makedirs(pv, exist_ok=True)
            info = emit_prevabs(cs, pv, name=tag, mesh_size=a.mesh_size)
            xml = info["xml"] if isinstance(info, dict) and "xml" in info else (tag + ".xml")
            for f in glob.glob(os.path.join(pv, "*.xml")) + glob.glob(os.path.join(pv, "*.dat")):
                shutil.copy(f, G["dirs"]["xml"])
            ok, tail = sh([G["prevabs"], "-i", os.path.basename(xml), "--vabs", "--hm"], cwd=pv,
                          timeout=a.pv_timeout)          # fail-fast so one bad station can't stall the batch
            sg = os.path.join(pv, tag + ".sg")
            if not os.path.exists(sg):
                raise RuntimeError("prevabs no .sg | " + (tail.strip().splitlines()[-1] if tail.strip() else ""))
            sg_out = os.path.join(G["dirs"]["sg"], tag + ".sg"); shutil.copy(sg, sg_out); rec["sg"] = True
            if G["want_2d"]:
                solid = os.path.join(G["dirs"]["2d_yaml"], tag + "_solid.yaml")
                ok2, tail2 = sh([sys.executable, CONVERT, sg_out, solid])
                rec["y2d"] = os.path.exists(solid)
                if not rec["y2d"]:
                    raise RuntimeError("convert_sg_to_yaml | " + (tail2.strip().splitlines()[-1] if tail2.strip() else ""))
    except Exception as e:
        rec["err"] = repr(e)[:200]
    rec["t"] = time.time() - t0
    print("[%-10s] r=%.4f chord=%5.2f webs=%d  sg=%s 2d=%s 1d=%s  [%.1fs]%s"
          % (tag, r, rec["chord"], rec["nweb"], rec["sg"], rec["y2d"], rec["y1d"], rec["t"],
             ("  ERR " + rec["err"]) if rec["err"] else ""), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="windIO -> per-station .xml -> .sg + 2-D + 1-D YAML")
    ap.add_argument("--windio", default=None, help="windIO v2 blade YAML (default: one in this dir, else bundled IEA-22)")
    ap.add_argument("--out", default=HERE, help="output directory (default: this script's directory)")
    ap.add_argument("--r", type=float, default=None, help="a SINGLE station r (0..1); overrides --stations")
    ap.add_argument("--stations", default=None, help="comma list of r; default = all windIO airfoil stations")
    ap.add_argument("--types", default="sg,1d,2d",
                    help="comma subset of {sg,1d,2d} to build (default all). 2d implies sg.")
    ap.add_argument("--mesh-size", type=float, default=0.01, help="chord-normalised PreVABS mesh size")
    ap.add_argument("--prefix", default="iea", help="output-file name prefix")
    ap.add_argument("--prevabs", default=None, help="path to prevabs binary (else auto/PREVABS_EXE)")
    ap.add_argument("--pv-timeout", dest="pv_timeout", type=int, default=360,
                    help="per-station PreVABS timeout in s (fail-fast; default 360)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many stations in PARALLEL (process pool; PreVABS is external, not JAX)")
    a = ap.parse_args()

    windio = a.windio or default_windio()
    types = {t.strip().lower() for t in a.types.split(",") if t.strip()}
    bad = types - {"sg", "1d", "2d"}
    if bad:
        sys.exit("--types must be a subset of {sg,1d,2d}; got extra %s" % bad)
    want_1d = "1d" in types
    want_2d = "2d" in types
    want_sg = ("sg" in types) or want_2d           # 2-D solid YAML is derived from the .sg

    if a.r is not None:
        stations = [a.r]
    elif a.stations:
        stations = [float(x) for x in a.stations.split(",")]
    else:
        stations = airfoil_stations(windio)

    prevabs = None
    if want_sg:
        prevabs = find_prevabs(a.prevabs)
        if prevabs is None:
            sys.exit("PreVABS binary not found (needed for sg/2d) -- set --prevabs or PREVABS_EXE")

    print("windIO   =", windio)
    print("out      =", a.out)
    print("types    =", sorted(types), "(build sg=%s 1d=%s 2d=%s)" % (want_sg, want_1d, want_2d))
    print("prevabs  =", prevabs)
    print("stations (%d) = %s" % (len(stations), ", ".join("%.4f" % s for s in stations)), flush=True)

    subs = ["xml", "sg", "2d_yaml", "1d_yaml"]
    dirs = {k: os.path.join(a.out, k) for k in subs}
    work = os.path.join(a.out, "_work")
    for d in list(dirs.values()) + [work]:
        os.makedirs(d, exist_ok=True)

    G.update(a=a, blade=load_blade(windio), dirs=dirs, work=work, prevabs=prevabs,
             want_1d=want_1d, want_sg=want_sg, want_2d=want_2d)
    if a.jobs and a.jobs > 1 and len(stations) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.jobs, len(stations))) as pool:   # PreVABS is external -> processes
            rows = pool.map(process_one, stations)
    else:
        rows = [process_one(r) for r in stations]

    need = [("sg", want_sg), ("y2d", want_2d), ("y1d", want_1d)]
    okc = sum(1 for x in rows if all(x[k] for k, w in need if w))
    print("\n%d/%d stations complete for types %s. Outputs in %s/{%s}"
          % (okc, len(rows), sorted(types), a.out, ", ".join(subs)), flush=True)


if __name__ == "__main__":
    main()
