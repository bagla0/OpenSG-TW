"""fig_spanwise.py -- (1) per-station buckling factor along the span, with the governing station and the
benchmark marked; (2) a process diagram of the OpenSG-RM -> FSM pipeline actually used.

No figure titles (the LaTeX caption is the title); legends outside the axes, vertical, on the right.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "fsm_scan.npz"))
st, lam = d["stations"].astype(int), d["lam1"]
BENCH, BENCH_ST = 1.04, 5
DZ = 100.0 / 29.0
gi = int(np.argmin(lam))

# ------------------------------------------------------------------ (1) spanwise lambda
fig, ax = plt.subplots(figsize=(9.2, 5.0))
ax.semilogy(st, lam, "o-", color="#0072B2", ms=5, lw=1.6, label=r"FSM $\lambda_1$ per station")
ax.axhline(BENCH, color="#D55E00", ls="--", lw=1.5,
           label=r"solid-SG benchmark  $\lambda\approx%.2f$" % BENCH)
ax.plot(st[gi], lam[gi], "*", color="#D55E00", ms=20, mec="k", mew=0.6, zorder=6,
        label=r"governing: st%d,  $\lambda_1=%.4f$" % (st[gi], lam[gi]))
ax.axvspan(4.5, 6.5, color="#009E73", alpha=0.12, zorder=0,
           label="segment 5 (stations 5$\\to$6)")
ax.annotate(r"st%d   $\lambda_1=%.4f$" % (st[gi], lam[gi]),
            xy=(st[gi], lam[gi]), xytext=(st[gi] + 2.6, lam[gi] * 3.0),
            arrowprops=dict(arrowstyle="->", color="k", lw=1.0), fontsize=10)
ax.annotate("load $\\to$ 0 at the tip,\nso $\\lambda$ diverges", xy=(28, 60), xytext=(20.5, 150),
            arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0), fontsize=9, color="0.35")
ax.set_xlabel("station index  (station $k$ at $z = k\\cdot100/29$ m)")
ax.set_ylabel(r"buckling load factor  $\lambda_1$")
ax.set_xticks(np.arange(st.min(), st.max() + 1, 2))
ax.grid(alpha=0.3, which="both")
sec = ax.secondary_xaxis("top", functions=(lambda s: s / 29.0, lambda r: r * 29.0))
sec.set_xlabel("$r/R$")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
fig.tight_layout()
p1 = os.path.join(HERE, "bar_urc_spanwise_lambda.png")
fig.savefig(p1, dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ (2) process diagram
fig, ax = plt.subplots(figsize=(10.5, 9.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 15.4); ax.axis("off")

BOX = dict(boxstyle="round,pad=0.12", linewidth=1.4)
STEPS = [
    ("#0072B2", "Shell_1DSG    1Dshell_k.yaml    (k = 0 ... 29)",
     "OML contour, 30 stations   |   62 nodes, 64 elements, webs for k >= 3\n"
     "verified OML: d_outer/t median 0.014   (a mid-surface contour would give 0.5)"),
    ("#009E73", "RM homogenization     build_rm_bundle(shell, ref='oml')",
     "=>  Timo 6x6,  warping V0/V1,  per-element ABD\n"
     "ref='oml' MUST be explicit - the yaml records no reference and the default is 'center'"),
    ("#E69F00", "beam load FF  from  bar_urc-k-t-0.in.glb",
     "line 5 = F1 M1 M2 M3 ,   line 6 = F2 F3     (so slot 3 of the six is M2, not F3)\n"
     "checked 3 ways: our own writer, dM2/dx1 = F3 to 1-3%, and the st15 hardcoded FF"),
    ("#CC79A7", "two-step dehomogenization     _macro_fields  +  _rm_shell_strain",
     "=>  8 RM shell strains at each element midpoint (xi = 0.5)\n"
     "validated at st15 vs VABS .SM: median s11 error 0.35% (cap centre) / 1.50% (circumferential)"),
    ("#56B4E9", "pre-buckling stress resultants     N = A eps + B kappa",
     "per element (N11, N22, N12)  -  the SAME ABD the FSM consumes,\n"
     "so N and ABD are consistent by construction (no layup mapping, no lofted mesh)"),
    ("#999999", "guards, before the eigenvalue is believed",
     "section equilibrium  -oint N11 z ds / M2 = 1.033        reduced axial  oint(A11-A12^2/A22)ds / EA = 1.000"),
    ("#0072B2", "FSM     solve_fsm_multi(nodes, cells, ABD, N, L, M)",
     "L = 100/29 = 3.448 m station spacing,  M harmonics,  branched (webbed) contour supported\n"
     "(K + lambda K_G) phi = 0,   K_G built from the membrane resultants only"),
    ("#D55E00", "lambda_1 per station   ->   minimum over the span",
     "governing st6: lambda_1 = 1.0468  (benchmark ~1.04),  m* = 4  =>  half-wave 0.862 m\n"
     "mode localized in layup_6 = carbon spar cap, between the two web junctions"),
]
h, gap = 1.35, 0.55
y = 15.4 - 0.65
for i, (c, head, sub) in enumerate(STEPS):
    y0 = y - h
    ax.add_patch(FancyBboxPatch((0.35, y0), 9.3, h, facecolor=c, alpha=0.16, edgecolor=c, **BOX))
    ax.text(5.0, y0 + h - 0.42, head, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(5.0, y0 + 0.44, sub, ha="center", va="center", fontsize=8.3, color="0.25")
    if i < len(STEPS) - 1:
        ax.add_patch(FancyArrowPatch((5.0, y0 - 0.03), (5.0, y0 - gap + 0.03),
                                     arrowstyle="-|>", mutation_scale=15, lw=1.5, color="0.35"))
    y = y0 - gap
fig.tight_layout()
p2 = os.path.join(HERE, "bar_urc_process.png")
fig.savefig(p2, dpi=200, bbox_inches="tight"); plt.close(fig)

print("wrote %s  (%.0f kB)" % (p1, os.path.getsize(p1) / 1024))
print("wrote %s  (%.0f kB)" % (p2, os.path.getsize(p2) / 1024))
print("\nper-station lambda_1:")
for s, l in zip(st, lam):
    mark = "   <== GOVERNING" if s == st[gi] else ""
    print("   st%02d  z=%6.2f m  r/R=%.3f   lam1=%10.4f%s" % (s, s * DZ, s / 29.0, l, mark))
