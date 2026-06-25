"""mh104 thickness sweep with the CCW-corrected shell.  For each factor f: regenerate the CCW mesh,
GATE on the e1/e2/e3 orientation match vs the solid (necessary check), then homogenize (JAX-Kirchhoff)
at OML (frac=0), center (0.5) and IML (1.0) and compare the diagonal Timo 6x6 to the solid where a
reference exists.  Run from the debug folder with the JAX env."""
import os
import subprocess
import sys
import numpy as np
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CC, "mh104_thickness_study", "results")
SOLY = os.path.join(CC, "mh104_thickness_study", "yaml_solid")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
PY = sys.executable


def _row(r):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)]


def _ir(r):
    return [int(float(v)) for v in (str(r[0]).split() if len(r) == 1 else r)]


def orient(p):
    d = yaml.safe_load(open(p)); nd = np.array([_row(r) for r in d["nodes"]])
    el = [np.array(_ir(r)) - 1 for r in d["elements"]]
    o = np.array([_row(r) for r in d["elementOrientations"]])
    ct = np.array([nd[e, :2].mean(0) for e in el])
    return ct, o[:, 0:3], o[:, 3:6], o[:, 6:9]


def check(solidyaml, shellyaml):
    Sc, _, Se2, Se3 = orient(solidyaml); Hc, He1, He2, He3 = orient(shellyaml)
    nn = (((Hc[:, None] - Sc[None]) ** 2).sum(-1)).argmin(1)
    e2 = (He2[:, :2] * Se2[nn, :2]).sum(1); e3 = (He3[:, :2] * Se3[nn, :2]).sum(1)
    return float(e2.mean()), float(e3.mean()), int((e2 < 0).sum()), int((e3 < -0.5).sum()), float(He1[:, 2].min())


for fi in (20, 40, 60, 80, 100):
    f = fi / 100.0
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    subprocess.run([PY, os.path.join(HERE, "build_ref_yaml.py"), "connect", "f=%.2f" % f],
                   cwd=HERE, capture_output=True)
    soly = os.path.join(SOLY, "solid_f%03d.yaml" % fi)
    e2m, e3m, nf2, nf3, e1z = check(soly, shy)
    ok = (e2m > 0.9 and e3m > 0.9 and nf2 == 0 and nf3 == 0 and abs(e1z - 1) < 1e-6)
    print("\n===== f=%.1f  ORIENTATION CHECK vs solid: e2.e2=%.3f e3.e3=%.3f (flips e2=%d e3=%d) e1_z>=%.3f  -> %s" % (
        f, e2m, e3m, nf2, nf3, e1z, "PASS" if ok else "FAIL"))
    cf = os.path.join(RES, "C6_solid_f%03d.txt" % fi)
    S = np.loadtxt(cf) if os.path.exists(cf) else None
    if S is not None:
        print("  diag %%diff vs solid:   EA     GA2    GA3    GJ     EI2    EI3")
    else:
        print("  (no solid 6x6 ref yet) abs diag:  EA        GA2       GA3       GJ        EI2       EI3")
    for frac, nm in ((0.0, "OML"), (0.5, "center"), (1.0, "IML")):
        C = np.asarray(solve_tw_from_yaml(shy, frac=frac)["Timo"]); C = 0.5 * (C + C.T)
        dg = [C[i, i] for i in range(6)]
        if S is not None:
            print("  %-7s " % nm + " ".join("%+6.1f" % (100 * (dg[i] - S[i, i]) / abs(S[i, i])) for i in range(6)))
        else:
            print("  %-7s " % nm + " ".join("%.3e" % dg[i] for i in range(6)))
