"""gen_16stations.py -- FULL cross-section dataset for the IEA-22-280-RWT wind blade at
ALL 16 windIO airfoil-definition stations (outer_shape.airfoils[].spanwise_position).

For each station r it generates, into full16/<tag>/ :
  1. shell_<tag>.yaml   -- 1-D shell SG YAML   (OpenSG_io build_cross_section -> emit_opensg_yaml,
                           default fraction=0.5 = laminate mid-surface, the RM/KL shell reference)
  2. iea_<tag>.sg       -- PreVABS VABS-input mesh (prevabs -i <xml> --vabs --hm, WITHOUT -e/--execute,
                           so PreVABS only WRITES the .sg and never invokes VABS -- in v2.1.0 --hm is
                           just the required analysis-mode selector; the user runs VABS locally for .K)
  3. solid_<tag>.yaml   -- 2-D solid OpenSG YAML (.sg -> convert_sg_to_yaml.py)
  4. C6_rm_<tag>.txt    -- RM 6x6 (the paper's method: ring_6dof(load_ring(shell)), 6-DOF drilling
                           Lagrange ring, mitc4_g23, contour-centroid reference)
  5. C6_solid_<tag>.txt -- JAX/FEniCS 2-D solid 6x6 (compute_timo_from_yaml), the on-server reference
                           (VABS is unavailable on this server; the .sg is emitted for local VABS)

Robust: every station is wrapped in try/except; a failure is recorded and the run continues.
Writes full16/summary.txt (human table) and full16/summary.npz (arrays).

    python gen_16stations.py                 # all 16 stations
    python gen_16stations.py --only r0534     # subset (comma list of tags)
    python gen_16stations.py --validate       # r0534 only + extra checks vs bundled references
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
for q in (MITC, REPO, IO, os.path.expanduser("~/OpenSG_io")):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax
jax.config.update("jax_enable_x64", True)

from opensg_io import load_blade, build_cross_section, emit_opensg_yaml, emit_prevabs
from xsec_5v6_master import load_ring, ring_6dof, LBL           # noqa: E402 (module runs its demo CASES on import)
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml

WINDIO = os.path.join(REPO, "examples", "data", "windio", "IEA-22-280-RWT.yaml")
CONVERT = os.path.join(IO, "scripts", "convert_sg_to_yaml.py")
_pv = sorted(glob.glob(os.path.expanduser("~/OpenSG_io/third_party/prevabs_bin/**/prevabs"), recursive=True))
PREVABS = _pv[-1] if _pv else None
OUT = os.path.join(HERE, "full16")
os.makedirs(OUT, exist_ok=True)

# the 16 windIO airfoil stations; tag = "r%04d" % round(r*1000) (matches the bundled iea_r0247/.../r0980)
STATIONS = [
    (0.0000, "r0000"), (0.0200, "r0020"), (0.0487, "r0049"), (0.0665, "r0067"),
    (0.0835, "r0084"), (0.1022, "r0102"), (0.1104, "r0110"), (0.1364, "r0136"),
    (0.1556, "r0156"), (0.1967, "r0197"), (0.2470, "r0247"), (0.3993, "r0399"),
    (0.5336, "r0534"), (0.7389, "r0739"), (0.9800, "r0980"), (1.0000, "r1000"),
]

SHELL_MS = 0.01     # shell-contour arc element size (chord-normalised); matches ex3/ex4/full_blade
SOLID_MS = 0.01     # PreVABS solid <mesh_size> (chord-normalised); ~bundled iea_*.sg resolution


def diag(C):
    return None if C is None else np.array([C[i, i] for i in range(6)])


def fmt_diag(C):
    if C is None:
        return "   --- (failed) ---"
    return "  ".join("%s=%.4g" % (LBL[i], C[i, i]) for i in range(6))


def run(cmd, cwd, timeout):
    """Run a subprocess, return (ok, tail-of-stderr)."""
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return p.returncode == 0, (p.stdout or "")[-400:]


def process(blade, r, tag):
    rec = dict(r=r, tag=tag, chord=float("nan"), nweb=0, nnodes_shell=0,
               prevabs_ok=False, sg=None, solid_yaml=None, shell_yaml=None,
               rm=None, solid=None, err="")
    sdir = os.path.join(OUT, tag)
    os.makedirs(sdir, exist_ok=True)

    # 1. cross-section + 1-D shell YAML (mid-surface reference, fraction=0.5 default)
    cs = build_cross_section(blade, r=r, mesh_size=SHELL_MS)
    rec["chord"] = float(cs["chord"])
    rec["nweb"] = len(cs["webs"])
    rec["nnodes_shell"] = len(cs["nodes"])
    shell_yaml = os.path.join(sdir, "shell_%s.yaml" % tag)
    emit_opensg_yaml(cs, shell_yaml)
    rec["shell_yaml"] = shell_yaml

    # 2. RM 6x6 (paper method: 6-DOF drilling-Lagrange ring)
    try:
        C6 = ring_6dof(load_ring(shell_yaml))
        rec["rm"] = np.asarray(C6)
        np.savetxt(os.path.join(sdir, "C6_rm_%s.txt" % tag), rec["rm"])
    except Exception as e:
        rec["err"] += "RM_FAIL:%r; " % (repr(e)[:90],)

    # 3. PreVABS XML -> .sg (VABS input mesh; --vabs only, NO --hm)
    pvdir = os.path.join(sdir, "prevabs")
    name = "iea_%s" % tag
    try:
        emit_prevabs(cs, pvdir, name=name, mesh_size=SOLID_MS)
        # PreVABS v2.1.0: --hm is the (required) analysis-MODE selector, --vabs the output FORMAT;
        # VABS is only actually run with -e/--execute (omitted here), so this only WRITES the .sg.
        ok, tail = run([PREVABS, "-i", name + ".xml", "--vabs", "--hm"], cwd=pvdir, timeout=1200)
        sg = os.path.join(pvdir, name + ".sg")
        if not (ok and os.path.exists(sg)):
            raise RuntimeError("prevabs no .sg: %s" % tail.strip().splitlines()[-1:])
        rec["prevabs_ok"] = True
        sg_dest = os.path.join(sdir, name + ".sg")
        shutil.copy(sg, sg_dest)
        rec["sg"] = sg_dest
    except Exception as e:
        rec["err"] += "PREVABS_FAIL:%s; " % (repr(e)[:120],)
        return rec

    # 4. .sg -> 2-D solid OpenSG YAML
    solid_yaml = os.path.join(sdir, "solid_%s.yaml" % tag)
    try:
        ok, tail = run([sys.executable, CONVERT, rec["sg"], solid_yaml], cwd=sdir, timeout=600)
        if not (ok and os.path.exists(solid_yaml)):
            raise RuntimeError("convert_sg_to_yaml failed: %s" % tail.strip().splitlines()[-1:])
        rec["solid_yaml"] = solid_yaml
    except Exception as e:
        rec["err"] += "CONVERT_FAIL:%s; " % (repr(e)[:120],)
        return rec

    # 5. 2-D solid 6x6 (JAX/FEniCS MSG, VABS-equivalent)
    try:
        S = np.asarray(compute_timo_from_yaml(solid_yaml, verbose=False))
        rec["solid"] = S
        np.savetxt(os.path.join(sdir, "C6_solid_%s.txt" % tag), S)
    except Exception as e:
        rec["err"] += "SOLIDHOMO_FAIL:%s; " % (repr(e)[:120],)
    return rec


def write_summary(rows):
    n = len(rows)
    RM = np.full((n, 6, 6), np.nan)
    SO = np.full((n, 6, 6), np.nan)
    for k, rec in enumerate(rows):
        if rec.get("rm") is not None:
            RM[k] = rec["rm"]
        if rec.get("solid") is not None:
            SO[k] = rec["solid"]
    np.savez(os.path.join(OUT, "summary.npz"),
             r=np.array([x["r"] for x in rows]),
             tags=np.array([x["tag"] for x in rows]),
             chord=np.array([x["chord"] for x in rows]),
             nweb=np.array([x["nweb"] for x in rows]),
             prevabs_ok=np.array([x["prevabs_ok"] for x in rows]),
             labels=np.array(LBL), rm=RM, solid=SO)

    lines = []
    lines.append("IEA-22-280-RWT -- cross-section dataset, 16 windIO airfoil stations")
    lines.append("shell: build_cross_section(ms=%.3f)+emit_opensg_yaml(frac=0.5)  |  "
                 "solid: emit_prevabs(ms=%.3f) -> prevabs --vabs -> convert_sg_to_yaml" % (SHELL_MS, SOLID_MS))
    lines.append("RM 6x6 = ring_6dof(load_ring(shell))   solid 6x6 = compute_timo_from_yaml(solid_yaml)")
    lines.append("VABS is NOT run here (no license on server); the .sg is emitted for local VABS.")
    lines.append("")
    hdr = "%-6s %7s %6s %5s %5s %-9s" % ("tag", "r", "chord", "webs", "PVok", "time_s")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for x in rows:
        lines.append("%-6s %7.4f %6.2f %5d %5s %8.1f"
                     % (x["tag"], x["r"], x["chord"], x["nweb"],
                        "yes" if x["prevabs_ok"] else "NO", x.get("t", float("nan"))))
    lines.append("")
    lines.append("=== RM 6x6 diagonal (EA, GA2, GA3, GJ, EI2, EI3) ===")
    for x in rows:
        lines.append("  %-6s r=%.4f  %s" % (x["tag"], x["r"], fmt_diag(x.get("rm"))))
    lines.append("")
    lines.append("=== 2-D solid 6x6 diagonal (EA, GA2, GA3, GJ, EI2, EI3) ===")
    for x in rows:
        lines.append("  %-6s r=%.4f  %s" % (x["tag"], x["r"], fmt_diag(x.get("solid"))))
    lines.append("")
    lines.append("=== RM diagonal %% error vs 2-D solid ===")
    for x in rows:
        C, S = x.get("rm"), x.get("solid")
        if C is None or S is None:
            lines.append("  %-6s r=%.4f   (missing RM or solid)" % (x["tag"], x["r"]))
            continue
        e = [100.0 * (C[i, i] - S[i, i]) / S[i, i] if abs(S[i, i]) > 0 else float("nan") for i in range(6)]
        lines.append("  %-6s r=%.4f  %s" % (x["tag"], x["r"],
                     "  ".join("%s%+7.2f" % (LBL[i], e[i]) for i in range(6))))
    lines.append("")
    lines.append("=== failures / notes ===")
    any_fail = False
    for x in rows:
        if x["err"]:
            any_fail = True
            lines.append("  %-6s: %s" % (x["tag"], x["err"]))
    if not any_fail:
        lines.append("  (none -- all stations produced .sg + shell YAML + solid YAML + both 6x6)")
    lines.append("")
    lines.append("=== files per station (full16/<tag>/) ===")
    for x in rows:
        lines.append("  %-6s: %s" % (x["tag"], ", ".join(
            os.path.basename(p) for p in (x.get("shell_yaml"), x.get("sg"), x.get("solid_yaml")) if p)))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)
    print("wrote", os.path.join(OUT, "summary.txt"), "and summary.npz")


def load_vabs_timo(path):
    L = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i + 1:]:
        p = ln.split()
        try:
            vals = [float(x) for x in p]
            ok = (len(p) == 6)
        except ValueError:
            ok = False
        if ok:
            rows.append(vals)
        if len(rows) == 6:
            break
    return np.array(rows)


def validate(blade):
    print("================= VALIDATION =================", flush=True)
    D2 = os.path.join(REPO, "examples", "data", "2d_yaml")
    IB = os.path.join(REPO, "examples", "data", "iea_blade")
    VABS = os.path.join(D2, "IEA_VABS")

    # (A) solid homogenizer reproduces the bundled C6_solid_r050.txt from iea22_r050_solid.yaml
    sy = os.path.join(D2, "iea22_r050_solid.yaml")
    ref = os.path.join(IB, "C6_solid_r050.txt")
    if os.path.exists(sy) and os.path.exists(ref):
        S = np.asarray(compute_timo_from_yaml(sy, verbose=False))
        R = np.loadtxt(ref)
        d = np.abs(S - R)
        rel = d / (np.abs(R) + np.abs(R).max() * 1e-6)
        print("(A) compute_timo_from_yaml(iea22_r050_solid.yaml) vs bundled C6_solid_r050.txt:")
        print("    diag mine :", "  ".join("%.5g" % S[i, i] for i in range(6)))
        print("    diag bund :", "  ".join("%.5g" % R[i, i] for i in range(6)))
        print("    max |rel diff| over 6x6 = %.3e  -> %s" % (rel.max(), "MATCH" if rel.max() < 1e-3 else "DIFF"))
    else:
        print("(A) skipped -- missing", sy, "or", ref)

    # (B) full generation at r0534, compare fresh solid 6x6 vs bundled VABS .K
    print("\n(B) full pipeline at r0534 (0.5336) ...", flush=True)
    t0 = time.time()
    rec = process(blade, 0.5336, "r0534")
    rec["t"] = time.time() - t0
    print("    prevabs_ok=%s  chord=%.3f  webs=%d  err=%s" % (rec["prevabs_ok"], rec["chord"], rec["nweb"], rec["err"] or "none"))
    print("    RM    diag:", fmt_diag(rec["rm"]))
    print("    solid diag:", fmt_diag(rec["solid"]))
    Kf = os.path.join(VABS, "iea_r0534.sg.K")
    if os.path.exists(Kf) and rec["solid"] is not None:
        K = 0.5 * (load_vabs_timo(Kf) + load_vabs_timo(Kf).T)
        e = [100.0 * (rec["solid"][i, i] - K[i, i]) / K[i, i] for i in range(6)]
        print("    VABS  diag:", "  ".join("%s=%.4g" % (LBL[i], K[i, i]) for i in range(6)))
        print("    my-solid vs bundled VABS .K diag %%err:",
              "  ".join("%s%+6.2f" % (LBL[i], e[i]) for i in range(6)))

    # (C) generated shell topology vs bundled shell_r0534.yaml (frac differs: bundle=OML, mine=mid)
    bundled = os.path.join(REPO, "examples", "data", "1d_yaml", "IEA", "shell_r0534.yaml")
    if os.path.exists(bundled) and rec["shell_yaml"]:
        import yaml as _y
        a = _y.safe_load(open(rec["shell_yaml"]))
        b = _y.safe_load(open(bundled))
        print("\n(C) shell topology  mine vs bundled shell_r0534.yaml:")
        print("    nodes  %d vs %d   elements %d vs %d   elem-sets %d vs %d"
              % (len(a["nodes"]), len(b["nodes"]), len(a["elements"]), len(b["elements"]),
                 len(a["sets"]["element"]), len(b["sets"]["element"])))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of tags, e.g. r0534,r0739")
    ap.add_argument("--validate", action="store_true", help="r0534 only + checks vs bundled refs")
    a = ap.parse_args()
    if PREVABS is None:
        sys.exit("prevabs binary not found under ~/OpenSG_io/third_party/prevabs_bin")
    print("prevabs =", PREVABS, flush=True)
    blade = load_blade(WINDIO)

    if a.validate:
        validate(blade)
        return

    todo = STATIONS
    if a.only:
        want = set(a.only.split(","))
        todo = [(r, t) for (r, t) in STATIONS if t in want]

    rows = []
    for r, tag in todo:
        t0 = time.time()
        try:
            rec = process(blade, r, tag)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec = dict(r=r, tag=tag, chord=float("nan"), nweb=0, prevabs_ok=False,
                       rm=None, solid=None, shell_yaml=None, sg=None, solid_yaml=None,
                       err="BUILD_FAIL:%s" % (repr(e)[:150],))
        rec["t"] = time.time() - t0
        rows.append(rec)
        print("[%s] r=%.4f chord=%.2f webs=%d PVok=%s  RM:%s | SOLID:%s  [%.1fs] %s"
              % (tag, rec["r"], rec["chord"], rec["nweb"], rec["prevabs_ok"],
                 "ok" if rec.get("rm") is not None else "FAIL",
                 "ok" if rec.get("solid") is not None else "FAIL", rec["t"],
                 ("ERR " + rec["err"]) if rec["err"] else ""), flush=True)

    write_summary(rows)


if __name__ == "__main__":
    main()
