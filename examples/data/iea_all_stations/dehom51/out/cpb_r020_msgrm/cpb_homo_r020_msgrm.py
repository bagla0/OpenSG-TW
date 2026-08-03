"""cpb_homo_r020_msgrm.py -- rerun of the CPB-paper r/R=0.2 HOMOGENIZATION with the
MSG (Yu-2002 variational-asymptotic) wall transverse-shear G replacing the Whitney
complementary-energy G in every wall laminate.

Reproduces, at the MSG-RM 8x8 wall law:
  (1) tab:iea020 -- RM shell vs the fixed 2-D solid (C6_solid_r020) at the OML
      laminate reference AND the centered reference, with deltas vs the PUBLISHED
      Whitney-G numbers;
  (2) the contour-refinement convergence sweep (figs/conv_iea_r020.png style).
Outputs (this folder): tab/iea020_msgrm.tex, dat/*.dat (6x6 + %err), figures/
conv_iea_r020_msgrm.png, data/homo_report.txt.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))                 # dehom51
ROOT = os.path.abspath(os.path.join(IEA, ".."))                       # iea_all_stations
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
TAPER = os.path.abspath(os.path.join(XSEC, "..", "..", "taper"))
for q in (XSEC, TAPER, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml as _yaml

from oml_ring import load_ring_ref, c6, derr
from xsec_5v6_master import load_solid
from msg_rm_plate import rm_plate_msg
from emit_abd import material_db_from_yaml

IB = os.path.abspath(os.path.join(XSEC, "..", "iea22_blade", "data"))
SHELL = os.path.join(IB, "shell_r020.yaml")
for d in ("tab", "dat", "figures", "data"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)

So = load_solid(os.path.join(IB, "C6_solid_r020.txt"))
dsh = _yaml.safe_load(open(SHELL))
mdb = material_db_from_yaml(dsh["materials"])


def msg_G_by(R, sections):
    """Replace every section's Whitney G with the MSG (VAM LS-projection) G."""
    G_by = list(R["G_by"])
    for si, sec in enumerate(sections):
        pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
        h = sum(p[1] for p in pl)
        rr = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl], [p[0] for p in pl],
                          mdb, fraction=0.5)
        if rr["G_msg"] is not None:
            G_by[si] = np.asarray(rr["G_msg"])
    R2 = dict(R)
    R2["G_by"] = G_by
    return R2


# ---------------- (1) the two-reference table ----------------
C_oml = c6(msg_G_by(load_ring_ref(SHELL, "oml"), dsh["sections"]))
C_cen = c6(msg_G_by(load_ring_ref(SHELL, "center"), dsh["sections"]))

ROWS = [("C^{b}_{11}", 0, 0), ("C^{b}_{15}", 0, 4), ("C^{b}_{16}", 0, 5),
        ("C^{b}_{22}", 1, 1), ("C^{b}_{33}", 2, 2), ("C^{b}_{34}", 2, 3),
        ("C^{b}_{44}", 3, 3), ("C^{b}_{55}", 4, 4), ("C^{b}_{56}", 4, 5),
        ("C^{b}_{66}", 5, 5)]
# published (Whitney-G) shell values from the CPB iea.tex, x1e9
PUB_OML = {"C^{b}_{11}": 27.96, "C^{b}_{15}": 1.624, "C^{b}_{16}": -71.46,
           "C^{b}_{22}": 0.7198, "C^{b}_{33}": 0.4070, "C^{b}_{34}": 0.8417,
           "C^{b}_{44}": 4.079, "C^{b}_{55}": 35.49, "C^{b}_{56}": -9.645,
           "C^{b}_{66}": 251.9}
PUB_CEN = {"C^{b}_{11}": 27.96, "C^{b}_{15}": 1.669, "C^{b}_{16}": -71.48,
           "C^{b}_{22}": 0.7230, "C^{b}_{33}": 0.4393, "C^{b}_{34}": 0.9249,
           "C^{b}_{44}": 4.525, "C^{b}_{55}": 38.00, "C^{b}_{56}": -9.968,
           "C^{b}_{66}": 252.3}

rep = ["=== CPB r/R=0.2 homogenization RERUN with the MSG (VAM) wall G ===",
       "shell yaml: %s" % SHELL,
       "%-12s %10s | %10s %7s (dWhit) | %10s %7s (dWhit)"
       % ("Cij", "solid", "OML", "%err", "center", "%err")]
for lab, i, j in ROWS:
    s = So[i, j] / 1e9
    vo = C_oml[i, j] / 1e9
    vc = C_cen[i, j] / 1e9
    eo = 100.0 * (vo - s) / abs(s)
    ec = 100.0 * (vc - s) / abs(s)
    dwo = 100.0 * (vo - PUB_OML[lab]) / abs(PUB_OML[lab])
    dwc = 100.0 * (vc - PUB_CEN[lab]) / abs(PUB_CEN[lab])
    rep.append("%-12s %10.4g | %10.4g %+7.2f (%+6.3f) | %10.4g %+7.2f (%+6.3f)"
               % (lab, s, vo, eo, dwo, vc, ec, dwc))
print("\n".join(rep), flush=True)

# 6x6 .dat + %err (house convention: entries below max|solid|/1e6 blanked)
np.savetxt(os.path.join(HERE, "dat", "C6_solid_r020.dat"), So, fmt="%14.6e")
np.savetxt(os.path.join(HERE, "dat", "C6_shell_oml_msgrm.dat"), C_oml, fmt="%14.6e")
np.savetxt(os.path.join(HERE, "dat", "C6_shell_center_msgrm.dat"), C_cen, fmt="%14.6e")
cut = np.abs(So).max() / 1e6
for tag, C in (("oml", C_oml), ("center", C_cen)):
    E = np.where(np.abs(So) > cut, 100.0 * (C - So) / np.where(np.abs(So) > cut, So, 1.0), 0.0)
    np.savetxt(os.path.join(HERE, "dat", "pcterr_%s_msgrm.dat" % tag), E, fmt="%10.3f")

# CPB-style table (cas-dc tabular*)
tex = [r"\begin{table}[width=.9\linewidth,cols=6,pos=h]",
       r"\caption{IEA-22 cross-section at $r/R=0.2$ with the MSG (variational-asymptotic)",
       r"wall transverse-shear stiffness: RM shell vs.\ the fixed 2-D solid at the OML",
       r"laminate reference and at the centered reference. Entries $\times10^{9}$ (SI).}",
       r"\label{tab:iea020msg}",
       r"\begin{tabular*}{\tblwidth}{@{}L R RR RR@{}}",
       r"\toprule",
       r"& & \multicolumn{2}{c}{OML reference} & \multicolumn{2}{c}{Centered reference}\\",
       r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
       r"$C^{b}_{ij}$ & Solid & Shell & \%err & Shell & \%err\\",
       r"\midrule"]
for lab, i, j in ROWS:
    s = So[i, j] / 1e9
    vo = C_oml[i, j] / 1e9
    vc = C_cen[i, j] / 1e9
    fmt = lambda v: ("%.4g" % v)
    tex.append(r"$%s$ & $%s$ & $%s$ & $%+.1f$ & $%s$ & $%+.1f$\\"
               % (lab, fmt(s), fmt(vo), 100.0 * (vo - s) / abs(s),
                  fmt(vc), 100.0 * (vc - s) / abs(s)))
tex += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
open(os.path.join(HERE, "tab", "iea020_msgrm.tex"), "w").write("\n".join(tex) + "\n")
print("wrote tab/iea020_msgrm.tex + dat/*.dat", flush=True)

# ---------------- (2) convergence sweep ----------------
from taper_common import WINDIO
from opensg_io.converter import load_blade, build_cross_section, emit_opensg_yaml

SCR = os.path.join(HERE, "data", "_conv_scr")
os.makedirs(SCR, exist_ok=True)
blade = load_blade(WINDIO)
MS = [0.06, 0.045, 0.03, 0.02, 0.015, 0.01]
LBL6 = [r"$EA$", r"$GA_2$", r"$GA_3$", r"$GJ$", r"$EI_2$", r"$EI_3$"]
rows, nn = [], []
for ms in MS:
    cs = build_cross_section(blade, 0.2, mesh_size=ms)
    sp = os.path.join(SCR, "shell_oml_ms%.3f.yaml" % ms)
    emit_opensg_yaml(cs, sp, fraction=0.0)
    dsp = _yaml.safe_load(open(sp))
    R = msg_G_by(load_ring_ref(sp, "oml"), dsp["sections"])
    C = c6(R)
    e = derr(C, So)
    rows.append(e)
    nn.append(len(R["rx"]))
    print("ms=%.3f  nnode=%-4d  diag %%err: %s"
          % (ms, nn[-1], "  ".join("%+6.2f" % v for v in e)), flush=True)
rows = np.array(rows)
nn = np.array(nn)
np.savez(os.path.join(HERE, "data", "conv_iea_r020_msgrm.npz"),
         mesh_size=np.array(MS), nnode=nn, diag_err=rows)

MK = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 12})
fig, ax = plt.subplots(figsize=(7.4, 5.0))
o = np.argsort(nn)
for k in range(6):
    ax.plot(nn[o], rows[o, k], "-" + MK[k], ms=7, lw=1.8, label=LBL6[k])
ax.set_xscale("log")
ax.axhline(0.0, color="0.5", lw=1.0)
ax.set_xlabel("contour nodes")
ax.set_ylabel("diagonal % error vs 2-D solid")
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "conv_iea_r020_msgrm.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("wrote figures/conv_iea_r020_msgrm.png", flush=True)
open(os.path.join(HERE, "data", "homo_report.txt"), "w").write("\n".join(rep) + "\n")
