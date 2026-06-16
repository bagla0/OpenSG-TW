"""
Build LaTeX tables, %-error plots and geometry/orientation figures for the
homogenization benchmark of MSG thin-walled shells (RM + Kirchhoff, centre & OML
references) against FEniCS-solid, for FOUR cases:
    tube  (isotropic, [45/-45]) -- thickness ratio h/R
    strip (isotropic, [45/-45]) -- thickness ratio h/W
Reads data/{solid,shell}_6x6.csv (tube) and data/strip_{solid,shell}_6x6.csv
(strip).  Writes report/tables.tex and report/figures/*.png, and prints a full
term-by-term verification summary.

Timoshenko order [ext, shear2, shear3, twist, bend2, bend3]; the six diagonals
are plotted/labelled with both the matrix index and the engineering meaning:
  C11 (EA)  C22 (GA2)  C33 (GA3)  C44 (GJ)  C55 (EI2)  C66 (EI3).
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, FancyArrow

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
REP = os.path.join(HERE, "report"); FIG = os.path.join(REP, "figures")
os.makedirs(FIG, exist_ok=True)

# six principal Timoshenko stiffnesses: (matrix index i, j) and (Cij, engineering)
DIAG = [(0, 0, "C_{11}", "EA"), (1, 1, "C_{22}", "GA_2"), (2, 2, "C_{33}", "GA_3"),
        (3, 3, "C_{44}", "GJ"), (4, 4, "C_{55}", "EI_2"), (5, 5, "C_{66}", "EI_3")]

# off-diagonal couplings to plot: the three the user asked for (C15,C16,C56 --
# geometry-driven, zero at the centroid for symmetric sections) PLUS the genuine
# antisymmetric-[45/-45] material couplings (C14 ext-twist, C25/C36 shear-bend).
COUPLINGS = [(0, 3, "C_{14}", r"\mathrm{ext\,-\,tw}"), (0, 4, "C_{15}", r"\mathrm{ext\,-\,}M_2"),
             (0, 5, "C_{16}", r"\mathrm{ext\,-\,}M_3"), (1, 4, "C_{25}", r"V_2\,\mathrm{-}\,M_2"),
             (2, 5, "C_{36}", r"V_3\,\mathrm{-}\,M_3"), (4, 5, "C_{56}", r"M_2\,\mathrm{-}\,M_3")]

# four shell "methods" -> distinct, well-separated colours; centre = solid line,
# OML = dashed; reference solid = the zero axis.
METHODS = [("RM, centre",  "center", "RM", "#1f77b4", "-",  "o"),
           ("KF, centre",  "center", "KF", "#2ca02c", "-",  "s"),
           ("RM, OML",     "OML",    "RM", "#ff7f0e", "--", "^"),
           ("KF, OML",     "OML",    "KF", "#d62728", "--", "v")]

CASES = [("tube",  "iso",   "h/R", "solid_6x6.csv",       "shell_6x6.csv"),
         ("tube",  "aniso", "h/R", "solid_6x6.csv",       "shell_6x6.csv"),
         ("strip", "iso",   "h/W", "strip_solid_6x6.csv", "strip_shell_6x6.csv"),
         ("strip", "aniso", "h/W", "strip_solid_6x6.csv", "strip_shell_6x6.csv")]

PRETTY = {("tube", "iso"): "Isotropic tube", ("tube", "aniso"): r"Anisotropic $[45/\!-\!45]$ tube",
          ("strip", "iso"): "Isotropic strip", ("strip", "aniso"): r"Anisotropic $[45/\!-\!45]$ strip"}


def load(csv, keycols):
    rows = {}
    with open(os.path.join(DATA, csv)) as f:
        hdr = f.readline().strip().split(",")
        ci = hdr.index("C11")
        for line in f:
            p = line.strip().split(",")
            key = tuple(p[:keycols])
            M = np.array([float(x) for x in p[ci:ci+36]]).reshape(6, 6)
            rows[key] = (float(p[1]), M)
    return rows


def case_data(solid_csv, shell_csv, mat):
    solid = load(solid_csv, 2)
    shell = load(shell_csv, 4)
    hrs = sorted({v[0] for k, v in solid.items() if k[0] == mat})
    return solid, shell, hrs


# ============================================================ plots
def plot_errors(geom, mat, xlab, solid, shell, hrs, fname):
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))
    fig.suptitle(f"{PRETTY[(geom, mat)]}: shell-vs-solid \\% error in the "
                 f"Timoshenko stiffness (solid = 0 line)", fontsize=13, fontweight="bold")
    for idx, (i, j, cij, eng) in enumerate(DIAG):
        a = ax.flat[idx]
        allvals, finite = {}, []
        for lbl, ref, mdl, col, ls, mk in METHODS:
            ys = []
            for hr in hrs:
                sv = solid[(mat, str(hr))][1][i, j]
                x = shell[(mat, str(hr), ref, mdl)][1][i, j]
                ys.append(100*(x-sv)/sv if abs(sv) > 1 else np.nan)
            ys = np.array(ys); allvals[lbl] = (ys, col, ls, mk)
            finite += [v for v in ys if np.isfinite(v) and abs(v) <= 100]
        # adaptive symmetric window: ignore reference artefacts > 100%
        M = max(3.0, 1.25*max(abs(v) for v in finite)) if finite else 30.0
        clipped = False
        for lbl, (ys, col, ls, mk) in allvals.items():
            a.plot(hrs, ys, color=col, ls=ls, marker=mk, ms=6, lw=1.8, label=lbl, clip_on=True)
            if np.any(np.abs(ys[np.isfinite(ys)]) > M):
                clipped = True
        a.set_ylim(-M, M)
        a.axhline(0, color="0.45", lw=1.0, zorder=0)
        a.set_title(f"${cij}$ (${eng}$)", fontsize=12)
        a.set_xlabel(f"${xlab}$"); a.set_ylabel("\\% error vs solid")
        a.grid(True, ls=":", alpha=0.6)
        if clipped:
            a.text(0.97, 0.04, "OML off-scale\n(parallel-axis; see table)", transform=a.transAxes,
                   ha="right", va="bottom", fontsize=7, style="italic", color="0.35")
    ax.flat[0].legend(fontsize=9, loc="best", framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


def plot_couplings(geom, mat, xlab, solid, shell, hrs, fname):
    """Normalized coupling coefficient |c_ij| = |C_ij|/sqrt(Cii*Cjj) vs thickness,
    shell methods against the solid (off-diagonal signs are axis-convention
    dependent, so the magnitude is the convention-independent comparison)."""
    def coef(M, i, j):
        d = np.sqrt(abs(M[i, i]*M[j, j]))
        return abs(M[i, j])/d if d > 0 else np.nan
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))
    fig.suptitle(f"{PRETTY[(geom, mat)]}: normalized coupling "
                 f"$|c_{{ij}}|=|C_{{ij}}|/\\sqrt{{C_{{ii}}C_{{jj}}}}$ "
                 f"(shell vs FEniCS-solid)", fontsize=13, fontweight="bold")
    for idx, (i, j, cij, eng) in enumerate(COUPLINGS):
        a = ax.flat[idx]
        sv = [coef(solid[(mat, str(hr))][1], i, j) for hr in hrs]
        a.plot(hrs, sv, color="k", ls="-", marker="D", ms=7, lw=2.4,
               label="FEniCS-solid", zorder=5)
        for lbl, ref, mdl, col, ls, mk in METHODS:
            ys = [coef(shell[(mat, str(hr), ref, mdl)][1], i, j) for hr in hrs]
            a.plot(hrs, ys, color=col, ls=ls, marker=mk, ms=6, lw=1.8, label=lbl)
        a.set_title(f"${cij}$ (${eng}$)", fontsize=12)
        a.set_xlabel(f"${xlab}$"); a.set_ylabel("$|c_{ij}|$")
        a.set_ylim(bottom=0.0); a.grid(True, ls=":", alpha=0.6)
    ax.flat[0].legend(fontsize=8.5, loc="best", framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


# ============================================================ geometry figures
def _ply_square(ax, x0, y0, s, ang_deg, color, label):
    rect = Rectangle((x0, y0), s, s, facecolor=color, edgecolor="k", lw=1.3)
    ax.add_patch(rect)
    th = np.deg2rad(ang_deg); dx, dy = np.cos(th), np.sin(th)
    cx, cy = x0 + s/2, y0 + s/2
    for c in np.linspace(-1.4, 1.4, 13):                       # fiber tows at the ply angle
        ox, oy = cx + (-dy)*c*s/2, cy + (dx)*c*s/2
        ln, = ax.plot([ox - 1.6*s*dx, ox + 1.6*s*dx], [oy - 1.6*s*dy, oy + 1.6*s*dy],
                      color="k", lw=0.9, alpha=0.6)
        ln.set_clip_path(rect)
    ax.text(x0 + s + 0.12*s, y0 + s/2, label, ha="left", va="center", fontsize=9)


def _layup_panel(ax):
    """Shared [+45/-45] layup orientation in the (e1 beam, e2 hoop/width) plane."""
    ax.set_title("Material layup orientation\n$[+45/\\!-\\!45]$", fontsize=11)
    _ply_square(ax, 0.0, 1.25, 1.0, 45.0, "#cfe3f7", "+45$^\\circ$ ply (outer / top)")
    _ply_square(ax, 0.0, 0.0,  1.0, -45.0, "#f7d9cf", "$-$45$^\\circ$ ply (inner / bottom)")
    ax.annotate("", xy=(1.35, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(1.40, 0.0, "$e_1$ (beam axis)", va="center", fontsize=9)
    ax.annotate("", xy=(0.0, 2.55), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.02, 2.60, "$e_2$ (hoop / width)", ha="left", fontsize=9)
    ax.set_xlim(-0.5, 2.9); ax.set_ylim(-0.45, 2.95); ax.set_aspect("equal"); ax.axis("off")


def _coarse_band_mesh(ax, kind, R=1.0, H=0.28, W=2.4):
    """Solid cross-section with the two ply bands + a coarse quad mesh."""
    if kind == "tube":
        ri, ro = R-H/2, R+H/2; rm = R
        for r0, r1, c in [(ri, rm, "#f7d9cf"), (rm, ro, "#cfe3f7")]:
            ax.add_patch(Wedge((0, 0), r1, 0, 360, width=r1-r0, facecolor=c,
                               edgecolor="none"))
        th = np.linspace(0, 2*np.pi, 49)
        for r in np.linspace(ri, ro, 5):
            ax.plot(r*np.cos(th), r*np.sin(th), color="0.5", lw=0.5)
        for t in np.linspace(0, 2*np.pi, 41)[:-1]:
            ax.plot([ri*np.cos(t), ro*np.cos(t)], [ri*np.sin(t), ro*np.sin(t)],
                    color="0.5", lw=0.5)
        ax.plot(R*np.cos(th), R*np.sin(th), "k--", lw=1.4)
        ax.text(0, 0, "mid-surface\nreference", ha="center", va="center", fontsize=8)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    else:
        for y0, y1, c in [(-H/2, 0.0, "#f7d9cf"), (0.0, H/2, "#cfe3f7")]:
            ax.add_patch(Rectangle((-W/2, y0), W, y1-y0, facecolor=c, edgecolor="none"))
        for y in np.linspace(-H/2, H/2, 5):
            ax.plot([-W/2, W/2], [y, y], color="0.5", lw=0.5)
        for x in np.linspace(-W/2, W/2, 25):
            ax.plot([x, x], [-H/2, H/2], color="0.5", lw=0.5)
        ax.plot([-W/2, W/2], [0, 0], "k--", lw=1.4)
        ax.text(0, H*0.9, "mid-surface reference", ha="center", fontsize=8)
        ax.set_xlim(-W/2-0.2, W/2+0.2); ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"FEniCS 2D-solid cross-section\n({'annulus' if kind=='tube' else 'rectangle'})",
                 fontsize=11)


def _genome_panel(ax, kind, R=1.0, H=0.28, W=2.4):
    """Shell genome: reference line + local frame e1,e2,e3."""
    if kind == "tube":
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(R*np.cos(th), R*np.sin(th), "k-", lw=2.0)
        ang = np.deg2rad(35); p = np.array([R*np.cos(ang), R*np.sin(ang)])
        tang = np.array([-np.sin(ang), np.cos(ang)]); norm = -np.array([np.cos(ang), np.sin(ang)])
        ax.annotate("", xy=p+0.42*tang, xytext=p, arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
        ax.annotate("", xy=p+0.42*norm, xytext=p, arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
        ax.text(*(p+0.5*tang), "$e_2$", color="#1f77b4", fontsize=11)
        ax.text(*(p+0.5*norm), "$e_3$", color="#d62728", fontsize=11)
        ax.plot(*p, "ko", ms=5); ax.text(p[0]+0.05, p[1]+0.05, "$e_1\\odot$", fontsize=10)
        ax.text(0, 0, "1D structure\ngenome\n(mid-surface)", ha="center", va="center", fontsize=8)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        ax.set_title("Shell idealization: circular genome", fontsize=11)
    else:
        ax.plot([-W/2, W/2], [0, 0], "k-", lw=2.0)
        p = np.array([0.35, 0.0])
        ax.annotate("", xy=p+np.array([0.5, 0]), xytext=p,
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
        ax.annotate("", xy=p+np.array([0, -0.42]), xytext=p,
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
        ax.text(p[0]+0.55, 0.04, "$e_2$ (width)", color="#1f77b4", fontsize=10)
        ax.text(p[0]+0.03, -0.5, "$e_3$ (thickness)", color="#d62728", fontsize=10)
        ax.plot(*p, "ko", ms=5); ax.text(p[0]+0.03, 0.08, "$e_1\\odot$ (beam)", fontsize=10)
        ax.text(0, 0.18, "1D structure genome (mid-surface)", ha="center", fontsize=8)
        ax.set_xlim(-W/2-0.2, W/2+0.2); ax.set_ylim(-0.6, 0.6)
        ax.set_title("Shell idealization: flat genome", fontsize=11)
    ax.set_aspect("equal"); ax.axis("off")


def geometry_figure(kind, fname):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    _genome_panel(ax[0], kind)
    _layup_panel(ax[1])
    _coarse_band_mesh(ax[2], kind)
    ttl = "Circular tube" if kind == "tube" else "Flat plate-strip"
    fig.suptitle(f"{ttl}: shell genome, $[45/\\!-\\!45]$ material orientation, and "
                 f"the matching 3D-solid cross-section", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


# ============================================================ tables
def fmt(x):
    return f"{x:.4e}".replace("e+0", "e+").replace("e-0", "e-")


def tables(allcase):
    tex = []
    for geom, mat, xlab, solid, shell, hrs in allcase:
        name = PRETTY[(geom, mat)]; tag = f"{geom}_{mat}"
        hmax = hrs[-1]
        # --- (1) absolute principal Timoshenko stiffness from the solid vs thickness
        tex += [r"\begin{table}[t]\centering",
                rf"\caption{{{name}: principal Timoshenko stiffnesses from the "
                rf"3D solid (SI units) vs ${xlab}$.}}",
                rf"\label{{tab:{tag}_props}}", r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lrrrrrr}\toprule",
                rf"${xlab}$ & $C_{{11}}$ ($EA$) & $C_{{22}}$ ($GA_2$) & "
                r"$C_{33}$ ($GA_3$) & $C_{44}$ ($GJ$) & $C_{55}$ ($EI_2$) & "
                r"$C_{66}$ ($EI_3$)\\\midrule"]
        for hr in hrs:
            M = solid[(mat, str(hr))][1]
            tex.append(f"{hr:.2f} & " + " & ".join(fmt(M[i, i]) for i in range(6)) + r"\\")
        tex += [r"\bottomrule\end{tabular}}", r"\end{table}", ""]

        # --- (2) %-error at the thickest section, by method x term
        tex += [r"\begin{table}[t]\centering",
                rf"\caption{{{name}: shell-vs-solid \% error at the thickest "
                rf"section (${xlab}={hmax:.2f}$).}}",
                rf"\label{{tab:{tag}_err}}", r"\resizebox{0.9\textwidth}{!}{%",
                r"\begin{tabular}{lrrrrrr}\toprule",
                r"method & $C_{11}$ & $C_{22}$ & $C_{33}$ & $C_{44}$ & "
                r"$C_{55}$ & $C_{66}$\\",
                r" & ($EA$) & ($GA_2$) & ($GA_3$) & ($GJ$) & ($EI_2$) & ($EI_3$)\\\midrule"]
        for lbl, ref, mdl, *_ in METHODS:
            cells = []
            for i, j, *_ in DIAG:
                sv = solid[(mat, str(hmax))][1][i, j]
                x = shell[(mat, str(hmax), ref, mdl)][1][i, j]
                cells.append(f"{100*(x-sv)/sv:+.2f}" if abs(sv) > 1 else "--")
            tex.append(f"{lbl} & " + " & ".join(cells) + r"\\")
        tex += [r"\bottomrule\end{tabular}}", r"\end{table}", ""]

        # --- (3) full 6x6 solid Timoshenko at the thickest section (exposes coupling)
        M = solid[(mat, str(hmax))][1]
        tex += [r"\begin{table}[t]\centering",
                rf"\caption{{{name}: full $6\times6$ Timoshenko stiffness from the "
                rf"3D solid at ${xlab}={hmax:.2f}$ (SI). Order "
                r"$[\,\text{ext},\,V_2,\,V_3,\,\text{tw},\,M_2,\,M_3\,]$; off-diagonal "
                r"couplings are negligible for these symmetric / balanced sections.}",
                rf"\label{{tab:{tag}_full}}", r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{rrrrrr}\toprule"]
        for i in range(6):
            tex.append(" & ".join(fmt(M[i, j]) for j in range(6)) +
                       (r"\\\midrule" if i == 0 else r"\\"))
        tex += [r"\bottomrule\end{tabular}}", r"\end{table}", ""]
    with open(os.path.join(REP, "tables.tex"), "w") as f:
        f.write("\n".join(tex))


def summary_table(allcase):
    """One compact table: centre-reference % error of the six principal
    stiffnesses at the THICKEST section, RM and KF, for all four cases."""
    t = [r"\begin{table}[h]\centering",
         r"\caption{Centre-reference shell-vs-solid \% error of the principal "
         r"Timoshenko stiffnesses at the thickest section ($h/R=h/W=0.20$). "
         r"$C_{33}$ ($GA_3$) of the strip is the 1D-shell-limited thickness-direction "
         r"shear (large error expected for both models); $C_{44}$ ($GJ$) shows RM "
         r"beating KF for the thick open strip.}",
         r"\label{tab:summary}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llrrrrrr}\toprule",
         r"case & model & $C_{11}$ & $C_{22}$ & $C_{33}$ & $C_{44}$ & $C_{55}$ & $C_{66}$\\",
         r" & & ($EA$) & ($GA_2$) & ($GA_3$) & ($GJ$) & ($EI_2$) & ($EI_3$)\\\midrule"]
    for geom, mat, xlab, solid, shell, hrs in allcase:
        hm = hrs[-1]; name = PRETTY[(geom, mat)]
        for mdl in ["RM", "KF"]:
            cells = []
            for i, j, *_ in DIAG:
                sv = solid[(mat, str(hm))][1][i, j]
                x = shell[(mat, str(hm), "center", mdl)][1][i, j]
                cells.append(f"{100*(x-sv)/sv:+.1f}" if abs(sv) > 1 else "--")
            lead = f"{name} & {mdl}" if mdl == "RM" else f" & {mdl}"
            t.append(lead + " & " + " & ".join(cells) + r"\\")
        t.append(r"\midrule")
    t[-1] = r"\bottomrule"
    t += [r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(REP, "summary_table.tex"), "w") as f:
        f.write("\n".join(t))


# ============================================================ main
def main():
    allcase = []
    for geom, mat, xlab, scsv, shcsv in CASES:
        solid, shell, hrs = case_data(scsv, shcsv, mat)
        allcase.append((geom, mat, xlab, solid, shell, hrs))
        plot_errors(geom, mat, xlab, solid, shell, hrs, f"err_{geom}_{mat}.png")
        plot_couplings(geom, mat, xlab, solid, shell, hrs, f"coupl_{geom}_{mat}.png")
    geometry_figure("tube", "geom_tube.png")
    geometry_figure("strip", "geom_strip.png")
    tables(allcase)
    summary_table(allcase)

    print("=== verification summary: % error vs solid (diagonal Timoshenko) ===")
    for geom, mat, xlab, solid, shell, hrs in allcase:
        print(f"\n## {geom}/{mat}  ({xlab})")
        print(f"{xlab:>6s}" + "".join(f"{eng:>9s}" for *_, eng in DIAG)
              + "   [RM-c / KF-c / RM-o / KF-o per term below]")
        for hr in hrs:
            line = f"{hr:6.2f}"
            for i, j, cij, eng in DIAG:
                sv = solid[(mat, str(hr))][1][i, j]
                rc = shell[(mat, str(hr), "center", "RM")][1][i, j]
                kc = shell[(mat, str(hr), "center", "KF")][1][i, j]
                ro = shell[(mat, str(hr), "OML", "RM")][1][i, j]
                ko = shell[(mat, str(hr), "OML", "KF")][1][i, j]
                pe = lambda x: 100*(x-sv)/sv if abs(sv) > 1 else float('nan')
                line += f" |{pe(rc):+5.1f}/{pe(kc):+5.1f}/{pe(ro):+5.1f}/{pe(ko):+5.1f}"
            print(line)
    print("\nwrote report/tables.tex, err_{tube,strip}_{iso,aniso}.png, "
          "geom_{tube,strip}.png")


if __name__ == "__main__":
    main()
