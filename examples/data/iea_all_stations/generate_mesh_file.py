'''
================================================================================================
 generate_mesh_file.py  --  PreVABS .xml  ->  OpenSG mesh files (.sg / 2-D solid / 1-D shell)
================================================================================================
The PreVABS **.xml is the common interface**.  Whatever produced it -- windIO
(obtain_input_cs_files.py) OR OpenFAST / NuMAD / a hand-written section -- this ONE script turns
any such .xml into the OpenSG homogenizer inputs.  It is a thin CLI over opensg_io.prevabs_xml:

      section.xml ──to_shell──► <name>_shell.yaml   (1-D shell SG YAML  -> RM homogenizer)
                 └──to_solid──► <name>.sg           (VABS input mesh; run VABS -> .K)
                                     └─convert──► <name>_solid.yaml  (2-D solid YAML -> JAX / FEniCSx)

  to_shell : parse_prevabs_xml(xml) -> emit_opensg_yaml           (no PreVABS; near-instant)
  to_solid : prevabs -i xml --vabs --hm -> .sg -> convert_sg_to_yaml.py -> 2-D solid YAML

OUTPUT  (one folder per type under --out, same layout obtain_input_cs_files.py writes)
  <out>/sg/       <name>.sg
  <out>/2d_yaml/  <name>_solid.yaml
  <out>/1d_yaml/  <name>_shell.yaml

ARGUMENTS
  xml              a single <name>.xml, OR a directory of .xml files (its materials.xml must sit
                   beside it -- emit/obtain always writes them together)
  --out DIR        output directory (default: this script's directory)
  --types LIST     comma subset of {sg,1d,2d} (default all).  2d implies sg; "--types 1d" runs
                   NO PreVABS (instant).
  --prevabs PATH   prevabs binary (default: auto-detect / $PREVABS_EXE)
  --jobs N         build this many xml in parallel (PreVABS is external, so processes)

EXAMPLES
  python generate_mesh_file.py xml/                       # every xml -> sg + 2d + 1d
  python generate_mesh_file.py xml/iea_r0247.xml          # one xml, all types
  python generate_mesh_file.py xml/ --types 2d            # 2-D solid YAML (+ .sg) only
  python generate_mesh_file.py xml/ --types 1d            # 1-D shell YAML only, no PreVABS
================================================================================================
'''
import argparse
import glob
import os
import shutil
import sys
import time

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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from opensg_io.prevabs_xml import to_shell, to_solid   # noqa: E402  (xml -> 1D / 2D+sg)


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


G = {}          # shared context for fork workers


def process_one(xml):
    a = G["a"]
    name = os.path.splitext(os.path.basename(xml))[0]
    rec = dict(name=name, sg=False, y2d=False, y1d=False, err="")
    t0 = time.time()
    try:
        if G["want_1d"]:
            shell = os.path.join(G["dirs"]["1d_yaml"], name + "_shell.yaml")
            to_shell(xml, shell); rec["y1d"] = os.path.exists(shell)
        if G["want_2d"]:                                             # to_solid: prevabs -> .sg -> 2-D yaml
            solid = os.path.join(G["dirs"]["2d_yaml"], name + "_solid.yaml")
            to_solid(xml, solid, G["prevabs"], timeout=a.pv_timeout)   # fail-fast: a degenerate tip won't hang the batch
            rec["y2d"] = os.path.exists(solid)
            sg = os.path.join(os.path.dirname(os.path.abspath(xml)), name + ".sg")
            if os.path.exists(sg):
                shutil.copy(sg, os.path.join(G["dirs"]["sg"], name + ".sg")); rec["sg"] = True
        elif G["want_sg"]:                                          # .sg only (no 2-D convert)
            import subprocess
            xdir = os.path.dirname(os.path.abspath(xml))
            subprocess.run([G["prevabs"], "-i", os.path.basename(xml), "--vabs", "--hm"],
                           cwd=xdir, check=True, timeout=a.pv_timeout)
            sg = os.path.join(xdir, name + ".sg")
            if os.path.exists(sg):
                shutil.copy(sg, os.path.join(G["dirs"]["sg"], name + ".sg")); rec["sg"] = True
    except Exception as e:
        rec["err"] = repr(e)[:200]
    rec["t"] = time.time() - t0
    print("[%-12s] sg=%s 2d=%s 1d=%s  [%.1fs]%s"
          % (name, rec["sg"], rec["y2d"], rec["y1d"], rec["t"],
             ("  ERR " + rec["err"]) if rec["err"] else ""), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="PreVABS .xml -> OpenSG .sg / 2-D solid / 1-D shell mesh files")
    ap.add_argument("xml", help="a single <name>.xml OR a directory of .xml files")
    ap.add_argument("--out", default=HERE, help="output directory (default: this script's directory)")
    ap.add_argument("--types", default="sg,1d,2d", help="comma subset of {sg,1d,2d} (default all). 2d implies sg.")
    ap.add_argument("--prevabs", default=None, help="prevabs binary (else auto / $PREVABS_EXE)")
    ap.add_argument("--pv-timeout", dest="pv_timeout", type=int, default=360)
    ap.add_argument("--jobs", type=int, default=1, help="build this many xml in parallel (processes)")
    a = ap.parse_args()

    xmls = sorted(glob.glob(os.path.join(a.xml, "*.xml"))) if os.path.isdir(a.xml) else [a.xml]
    xmls = [x for x in xmls if not os.path.basename(x).endswith("materials.xml")]
    if not xmls:
        sys.exit("no .xml found at %s" % a.xml)
    types = {t.strip().lower() for t in a.types.split(",") if t.strip()}
    bad = types - {"sg", "1d", "2d"}
    if bad:
        sys.exit("--types must be a subset of {sg,1d,2d}; got extra %s" % bad)
    want_1d = "1d" in types
    want_2d = "2d" in types
    want_sg = ("sg" in types) or want_2d

    prevabs = find_prevabs(a.prevabs) if want_sg else None
    if want_sg and prevabs is None:
        sys.exit("PreVABS binary not found (needed for sg/2d) -- set --prevabs or PREVABS_EXE")

    dirs = {k: os.path.join(a.out, k) for k in ("sg", "2d_yaml", "1d_yaml")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    print("xml (%d) -> types %s  (out=%s, prevabs=%s)" % (len(xmls), sorted(types), a.out, prevabs), flush=True)

    G.update(a=a, dirs=dirs, prevabs=prevabs, want_1d=want_1d, want_2d=want_2d, want_sg=want_sg)
    if a.jobs and a.jobs > 1 and len(xmls) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.jobs, len(xmls))) as pool:
            rows = pool.map(process_one, xmls)
    else:
        rows = [process_one(x) for x in xmls]

    need = [("sg", want_sg), ("y2d", want_2d), ("y1d", want_1d)]
    okc = sum(1 for x in rows if all(x[k] for k, w in need if w))
    print("\n%d/%d xml complete for types %s -> %s/{sg,2d_yaml,1d_yaml}"
          % (okc, len(rows), sorted(types), a.out), flush=True)


if __name__ == "__main__":
    main()
