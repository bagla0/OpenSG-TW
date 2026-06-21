"""Full mh104 shell deliverable (JAX-Kirchhoff, CCW, k22=0).  For every thickness factor
f in {0.1,0.2,0.4,0.6,0.8,1.0}: regenerate the CCW mesh, write the e1/e2/e3 orientation image and the
solid-vs-shell orientation comparison (where a solid YAML exists), homogenize at OML/center/IML, and
save each full 6x6 to results/ plus a master diagonal summary."""
import os
import subprocess
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.msg_hermite import solve_tw_from_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
RES = os.path.join(STUDY, "results")
FIG = os.path.join(STUDY, "figures"); os.makedirs(FIG, exist_ok=True)
SOLY = os.path.join(STUDY, "yaml_solid")
lab = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
PY = sys.executable
FS = [10, 20, 40, 60, 80, 100]
REFS = [(0.0, "OML"), (0.5, "center"), (1.0, "IML")]

summ = []
for fi in FS:
    f = fi / 100.0
    subprocess.run([PY, os.path.join(HERE, "build_ref_yaml.py"), "connect", "f=%.2f" % f],
                   cwd=HERE, capture_output=True)
    shy = os.path.join(HERE, "shell_ref_f%03d_connect.yaml" % fi)
    subprocess.run([PY, os.path.join(HERE, "plot_orientation.py"), shy,
                    os.path.join(FIG, "orient_shell_f%03d.png" % fi)], capture_output=True)
    soly = os.path.join(SOLY, "solid_f%03d.yaml" % fi)
    if os.path.exists(soly):
        subprocess.run([PY, os.path.join(HERE, "compare_orient_solid_shell.py"), soly, shy,
                        os.path.join(FIG, "orient_cmp_f%03d" % fi)], capture_output=True)
    for frac, ref in REFS:
        C = np.asarray(solve_tw_from_yaml(shy, frac=frac)["Timo"]); C = 0.5 * (C + C.T)
        np.savetxt(os.path.join(RES, "C6_shell_jax_%s_f%03d.txt" % (ref, fi)), C,
                   header="mh104 JAX-Kirchhoff CCW k22=0  f=%.2f  ref=%s  order[EA,GA2,GA3,GJ,EI2,EI3]" % (f, ref))
        summ.append((fi, ref, [float(C[i, i]) for i in range(6)]))
    print("f=%.2f done" % f, flush=True)

with open(os.path.join(RES, "timo_shell_jax_summary.txt"), "w") as fh:
    fh.write("# mh104 JAX-Kirchhoff (CCW, k22=0) shell Timoshenko DIAGONAL -- all f, all refs\n")
    fh.write("# order: EA GA2 GA3 GJ EI2 EI3\n")
    fh.write("%-6s %-7s %s\n" % ("f", "ref", "  ".join("%-11s" % l for l in lab)))
    for fi, ref, dg in summ:
        fh.write("%-6.2f %-7s %s\n" % (fi / 100.0, ref, "  ".join("%.4e" % x for x in dg)))
print("WROTE results/timo_shell_jax_summary.txt + per-case C6 + figures/", flush=True)
