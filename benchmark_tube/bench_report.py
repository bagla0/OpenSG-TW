"""
Build LaTeX tables + plots for the tube homogenization benchmark:
shell (RM, Kirchhoff; centre & OML references) vs FEniCS-solid, over h/R.
Reads data/solid_6x6.csv and data/shell_6x6.csv; writes report/tables.tex and
report/figures/*.png.  Also prints a verification summary.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
REP = os.path.join(HERE, "report"); FIG = os.path.join(REP, "figures")
os.makedirs(FIG, exist_ok=True)
R = 1.0
ISO_E, ISO_NU = 70e9, 0.3
ISO_G = ISO_E/(2*(1+ISO_NU))
TERMS = {"iso":   [("EA", 0, 0), ("GA", 1, 1), ("GJ", 3, 3), ("EI", 4, 4)],
         "aniso": [("EA", 0, 0), ("GA", 1, 1), ("GJ", 3, 3), ("EI", 4, 4),
                   ("ext-twist", 0, 3)]}


def load(csv, keycols):
    rows = {}
    with open(os.path.join(DATA, csv)) as f:
        hdr = f.readline().strip().split(",")
        ci = hdr.index("C11")
        for line in f:
            p = line.strip().split(",")
            key = tuple(p[:keycols])
            M = np.array([float(x) for x in p[ci:ci+36]]).reshape(6, 6)
            rows[key] = (float(p[1]), M)            # (hr, 6x6)
    return rows


def iso_analytic(hr, term):
    H = hr*R; ri, ro = R-H/2, R+H/2
    A = np.pi*(ro**2-ri**2); I = np.pi/4*(ro**4-ri**4); J = np.pi/2*(ro**4-ri**4)
    return {"EA": ISO_E*A, "GA": ISO_G*A*0.5, "GJ": ISO_G*J, "EI": ISO_E*I}[term]


def main():
    solid = load("solid_6x6.csv", 2)               # key (material, hr)
    shell = load("shell_6x6.csv", 4)               # key (material, hr, ref, model)
    HRS = sorted({v[0] for k, v in solid.items() if k[0] == "iso"})

    tex = []; summary = []
    for mat in ["iso", "aniso"]:
        for tname, i, j in TERMS[mat]:
            tex.append(f"\\begin{{table}}[t]\\centering")
            tex.append(f"\\caption{{{'Isotropic' if mat=='iso' else 'Anisotropic [45/-45]'} "
                       f"tube: {tname} vs $h/R$ (shell vs FEniCS-solid; \\% relative to solid).}}")
            tex.append(f"\\label{{tab:{mat}_{tname.replace('-','')}}}")
            spec = "lrrrrrr" if mat == "iso" else "lrrrrr"
            tex.append(f"\\begin{{tabular}}{{{spec}}}\\toprule")
            extra = " & analytic\\%" if mat == "iso" else ""
            tex.append(f"$h/R$ & solid & RM-ctr\\% & KF-ctr\\% & RM-OML\\% & KF-OML\\%{extra}\\\\\\midrule")
            for hr in HRS:
                sv = solid[(mat, str(hr))][1][i, j]
                rc = shell[(mat, str(hr), "center", "RM")][1][i, j]
                kc = shell[(mat, str(hr), "center", "KF")][1][i, j]
                ro_ = shell[(mat, str(hr), "OML", "RM")][1][i, j]
                ko = shell[(mat, str(hr), "OML", "KF")][1][i, j]
                pe = lambda x: 100*(x-sv)/sv if abs(sv) > 1 else float('nan')
                ex = ""
                if mat == "iso":
                    av = iso_analytic(hr, tname); ex = f" & {100*(av-sv)/sv:+.2f}"
                tex.append(f"{hr:.2f} & {sv:.4e} & {pe(rc):+.2f} & {pe(kc):+.2f} & "
                           f"{pe(ro_):+.2f} & {pe(ko):+.2f}{ex}\\\\")
                summary.append((mat, tname, hr, sv, pe(rc), pe(kc), pe(ro_), pe(ko)))
            tex.append("\\bottomrule\\end{tabular}\\end{table}")
            tex.append("")
    with open(os.path.join(REP, "tables.tex"), "w") as f:
        f.write("\n".join(tex))

    # plots: % error vs h/R, per material, 2x2 (or 2x3) stiffness panels
    for mat in ["iso", "aniso"]:
        terms = TERMS[mat]; nt = len(terms)
        nc = 3 if nt > 4 else 2; nr = int(np.ceil(nt/nc))
        fig, ax = plt.subplots(nr, nc, figsize=(5.2*nc, 4.2*nr), squeeze=False)
        fig.suptitle(f"{'Isotropic' if mat=='iso' else 'Anisotropic [45/-45]'} tube: "
                     "shell vs FEniCS-solid (\\% error)", fontweight="bold")
        for idx, (tname, i, j) in enumerate(terms):
            a = ax.flat[idx]
            for lbl, ref, mdl, fmt in [("RM, centre", "center", "RM", "b-o"),
                                       ("KF, centre", "center", "KF", "c--s"),
                                       ("RM, OML", "OML", "RM", "r-^"),
                                       ("KF, OML", "OML", "KF", "m--v")]:
                ys = []
                for hr in HRS:
                    sv = solid[(mat, str(hr))][1][i, j]
                    x = shell[(mat, str(hr), ref, mdl)][1][i, j]
                    ys.append(100*(x-sv)/sv if abs(sv) > 1 else np.nan)
                a.plot(HRS, ys, fmt, ms=5, label=lbl)
            a.axhline(0, color="0.6", lw=0.8); a.set_title(tname)
            a.set_xlabel("$h/R$"); a.set_ylabel("\\% error vs solid")
            a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=7)
        for k in range(nt, nr*nc): ax.flat[k].axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(FIG, f"err_{mat}.png"), dpi=150); plt.close(fig)

        # absolute stiffness vs h/R (centre ref), log-log, shell vs solid
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for idx, (tname, i, j) in enumerate(terms[:4]):
            a = ax.flat[idx]
            sv = [solid[(mat, str(hr))][1][i, j] for hr in HRS]
            rc = [shell[(mat, str(hr), "center", "RM")][1][i, j] for hr in HRS]
            kc = [shell[(mat, str(hr), "center", "KF")][1][i, j] for hr in HRS]
            a.loglog(HRS, np.abs(sv), "g--^", ms=6, label="FEniCS-solid")
            a.loglog(HRS, np.abs(rc), "b-o", ms=4, label="MSG-TW RM (ctr)")
            a.loglog(HRS, np.abs(kc), "c:s", ms=4, label="MSG-TW KF (ctr)")
            a.set_title(tname); a.set_xlabel("$h/R$"); a.set_ylabel(tname)
            a.grid(True, which="both", ls=":", alpha=0.5); a.legend(fontsize=8)
        fig.suptitle(f"{'Isotropic' if mat=='iso' else 'Anisotropic [45/-45]'} tube: "
                     "absolute stiffness vs $h/R$ (centre ref)", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(FIG, f"abs_{mat}.png"), dpi=150); plt.close(fig)

    print("=== verification summary: % error vs solid ===")
    print(f"{'mat':6s}{'term':10s}{'h/R':>6s}{'solid':>12s}{'RMctr':>8s}{'KFctr':>8s}{'RMoml':>8s}{'KFoml':>8s}")
    for mat, tname, hr, sv, rc, kc, ro_, ko in summary:
        print(f"{mat:6s}{tname:10s}{hr:6.2f}{sv:12.3e}{rc:8.2f}{kc:8.2f}{ro_:8.2f}{ko:8.2f}")
    print("\nwrote report/tables.tex and report/figures/{err,abs}_{iso,aniso}.png")


if __name__ == "__main__":
    main()
