"""emit_r020_dehom_paper.py -- IEA r=0.2 dehomogenization paper assets (replaces st15):
  figures/r020_section_paths.png  2-D solid by material + the two OML dehom paths
  tab_rm/r020_homo.tex            RM ring Timoshenko 6x6 vs VABS .K (all nonzero Cij)
  figures/r020_disp.png           VABS .U local displacement recovery (warping + deformed)
The stress-recovery figures come from dehom_r020_figs.py.
"""
import os, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
from oml_ring import load_ring_ref, c6

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
IB = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper", "iea22_blade", "data"))
SOLID = os.path.join(D2, "iea_r020_solid.yaml")
SHELL = os.path.join(IB, "shell_r020.yaml")
KF = os.path.join(D2, "iea_r020.sg.K")
UF = os.path.join(D2, "iea_r020.sg.U")
TEX = os.path.join(HERE, "results", "tex_rm"); os.makedirs(TEX, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
NAMES = {1: "Gelcoat", 2: "Glass triax", 3: "Foam", 4: "Carbon", 5: "Glass uniax", 6: "Glass biax"}
FILL = {1: "#bbbbbb", 2: "#9ec7e8", 3: "#f0a3a3", 4: "#d9c69a", 5: "#c8b3de", 6: "#f6b78b"}


def rowf(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


ds = yaml.safe_load(open(SOLID))
nodes = np.array([rowf(n)[:2] for n in ds["nodes"]])
tris = [[int(round(x)) - 1 for x in rowf(e)] for e in ds["elements"]]
mat = np.zeros(len(tris), int)
for grp in ds["sets"]["element"]:
    mi = int("".join(c for c in grp["name"] if c.isdigit()))
    for lab in grp["labels"]:
        mat[int(lab) - 1] = mi
cap = np.loadtxt(os.path.join(D2, "solid.lp_sparcap_center_thickness_r020.coords"))[:, :2]
circ = np.loadtxt(os.path.join(D2, "solid.circumferential_r020.coords"))[:, :2]

# ---------------- section + paths figure ----------------
polys = [nodes[t] for t in tris]
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.add_collection(PolyCollection(polys, facecolors=[FILL[mat[k]] for k in range(len(tris))],
                                 edgecolors="none"))
ax.plot(circ[:, 0], circ[:, 1], "-", color="#1f77b4", lw=2.0, label="circumferential path (LP surface)")
ax.plot(cap[:, 0], cap[:, 1], "-o", color="#d62728", ms=3, lw=2.0, label="spar-cap through-thickness")
ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
h = [Patch(facecolor=FILL[m], edgecolor="0.4", label=NAMES[m]) for m in sorted(set(mat))]
h += [plt.Line2D([], [], color="#1f77b4", lw=2, label="circumferential (LP)"),
      plt.Line2D([], [], color="#d62728", lw=2, marker="o", ms=4, label="cap through-thickness")]
ax.legend(handles=h, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(os.path.join(FIG, "r020_section_paths.png"), dpi=170, bbox_inches="tight")
plt.close(fig); print("wrote r020_section_paths.png")

# ---------------- homogenization table (RM ring vs VABS .K) ----------------
def load_vabs_timo(path):
    L = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i + 1:]:
        p = ln.split()
        try:
            [float(x) for x in p]; ok = (len(p) == 6)
        except ValueError:
            ok = False
        if ok:
            rows.append([float(x) for x in p])
        if len(rows) == 6:
            break
    return np.array(rows)

K = 0.5 * (load_vabs_timo(KF) + load_vabs_timo(KF).T)
RM = c6(load_ring_ref(SHELL, "oml"))
gmax = max(abs(K[k, k]) for k in range(6))
NZ = [(i, j) for i in range(6) for j in range(i, 6) if i == j or abs(K[i, j]) >= 1e-3 * gmax]
DIAG = {0: "EA", 1: "GA_2", 2: "GA_3", 3: "GJ", 4: "EI_2", 5: "EI_3"}
lines = [r"\begin{table}[htpb]\centering\small",
         r"\caption{IEA r$=0.2$ root Timoshenko $6\times6$: every nonzero $C_{ij}$ of the RM shell "
         r"(the $C^0$ MITC-$\gamma_{23}$ ring, OML) vs.\ the VABS $.\mathrm{K}$ (2-D solid). VABS order "
         r"$[1\!=\!\mathrm{ext},2,3\!=\!\mathrm{shear},4\!=\!\mathrm{twist},5,6\!=\!\mathrm{bend}]$.}"
         r"\label{tab:r020_homo}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"$C_{ij}$ & VABS $.\mathrm{K}$ & RM shell & \%\,err \\", r"\midrule"]
for (i, j) in NZ:
    nm = "C_{%d%d}" % (i + 1, j + 1) + ("\\,(%s)" % DIAG[i] if i == j else "")
    v = K[i, j]
    lines.append("$%s$ & $%.3e$ & $%.3e$ & $%+.1f$ \\\\" % (nm, v, RM[i, j], 100 * (RM[i, j] - v) / v))
fro = np.linalg.norm(RM - K) / np.linalg.norm(K) * 100
lines += [r"\midrule", r"\multicolumn{4}{l}{\small full-$6\times6$ Frobenius error $=%.1f\%%$} \\" % fro,
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TEX, "r020_homo.tex"), "w").write("\n".join(lines))
print("wrote r020_homo.tex (%d nonzero terms, Frobenius %.2f%%)" % (len(NZ), fro))

# ---------------- local displacement (VABS .U) ----------------
U = np.loadtxt(UF)                                   # id y2 y3 u1 u2 u3
uxy = U[:, 1:3]; u1 = U[:, 3]; uip = U[:, 4:6]       # axial warping u1, in-plane (u2,u3)
sc = 0.15 * (np.ptp(uxy, 0).max()) / (np.linalg.norm(uip, axis=1).max() + 1e-30)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
s0 = ax[0].scatter(uxy[:, 0], uxy[:, 1], c=u1 * 1e3, s=2, cmap="coolwarm", linewidths=0)
ax[0].set_title(r"axial warping $u_1$ (mm)", fontsize=10); fig.colorbar(s0, ax=ax[0], shrink=0.8)
mag = np.linalg.norm(uip, axis=1) * 1e3
s1 = ax[1].scatter(uxy[:, 0] + sc * uip[:, 0], uxy[:, 1] + sc * uip[:, 1], c=mag, s=2,
                   cmap="viridis", linewidths=0)
ax[1].set_title(r"in-plane displacement (deformed $\times%.0f$, $|u_{23}|$ mm)" % sc, fontsize=10)
fig.colorbar(s1, ax=ax[1], shrink=0.8)
for a in ax:
    a.set_aspect("equal"); a.axis("off")
fig.suptitle("IEA r=0.2 local displacement recovered by dehomogenization (VABS $.\\mathrm{U}$)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(FIG, "r020_disp.png"), dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote r020_disp.png  (u1 range %.3f..%.3f mm, |u23| max %.3f mm)"
      % (u1.min() * 1e3, u1.max() * 1e3, mag.max()))
