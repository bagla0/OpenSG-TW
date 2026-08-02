"""plot_fig5.py -- the Fig.-5 plot for Nayak Ex.1: OpenSG-RM vs the
ANALYTIC benchmark (Reismann-Lee-type Mindlin Navier modal solution,
computed here) for the isotropic plate under the suddenly applied center
patch load.

Axes exactly as the paper's Fig. 5:
    w_bar = w E a h / (q b^3)      vs      t_bar = (t/b) sqrt(E/rho)
(with the nondimensional units of the deck, w_bar = 0.2828 w and
t_bar = t).  Analytic peak: w_bar = 3.21 at t_bar = 3.94 (static 1.61,
so the held step doubles it, as it must).

Run:  python examples/opensg-rm_dynamic/ex1/plot_fig5.py
"""
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AX, BY, H = 1.414, 1.0, 0.2
EMOD, RHO, QP = 1.0, 1.0, 1.0
DT = 0.045


def node_history(dat_path, nset):
    rows = []
    with open(dat_path, errors="replace") as f:
        lines = f.read().splitlines()
    active, labels_seen = False, False
    for ln in lines:
        if "NODE SET %s" % nset in ln and "TABLE IS PRINTED" in ln:
            active, labels_seen = True, False
            continue
        toks = ln.split()
        if not toks:
            continue
        if active and not labels_seen:
            if toks[0] == "NODE":
                labels_seen = True
            continue
        if active and labels_seen and re.fullmatch(r"\d+", toks[0]):
            vals = []
            for t in toks[1:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if len(vals) >= 3:
                rows.append(vals[2])
            active = False
    return np.array(rows)


def mindlin_analytic(tmax=9.0, npts=2001):
    """Reismann-Lee-type benchmark: SS Mindlin plate, Navier modal
    transient for the held step patch load (kappa = 5/6, rotary inertia
    included)."""
    NU, KAP = 0.30, 5.0 / 6.0
    D = EMOD * H ** 3 / (12 * (1 - NU ** 2))
    G = EMOD / (2 * (1 + NU))
    C = 0.4 * AX
    t = np.linspace(0, tmax, npts)
    w = np.zeros_like(t)
    for m in range(1, 30, 2):
        for n2 in range(1, 30, 2):
            qmn = (16 * QP / (np.pi ** 2 * m * n2)
                   * np.sin(m * np.pi / 2) * np.sin(n2 * np.pi / 2)
                   * np.sin(m * np.pi * C / (2 * AX))
                   * np.sin(n2 * np.pi * C / (2 * BY)))
            lam = np.pi ** 2 * ((m / AX) ** 2 + (n2 / BY) ** 2)
            wst = qmn / (D * lam ** 2) * (1 + D * lam / (KAP * G * H))
            meff = RHO * H * (1 + H ** 2 * lam / 12.0)
            om = np.sqrt((qmn / wst) / meff)
            w += (wst * np.sin(m * np.pi / 2) * np.sin(n2 * np.pi / 2)
                  * (1 - np.cos(om * t)))
    return t / BY * np.sqrt(EMOD / RHO), w * EMOD * AX * H / (QP * BY ** 3)


def main():
    dat = os.path.join(HERE, "Abaqus_results", "ex1_RM_fig5.dat")
    w = node_history(dat, "NCEN")
    t = DT * np.arange(1, len(w) + 1)
    wbar = w * EMOD * AX * H / (QP * BY ** 3)
    tbar = (t / BY) * np.sqrt(EMOD / RHO)
    ta, wa = mindlin_analytic()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ta, wa, "--", color="k", lw=1.6,
            label="Analytic (Reismann-Lee type\nMindlin modal solution)")
    ax.plot(np.concatenate([[0], tbar]), np.concatenate([[0], wbar]),
            "-s", color="#ff7f0e", lw=1.4, ms=4, mfc="none", mew=1.1,
            markevery=8, label="OpenSG-RM\n(Abaqus S4 + MSG ABDG)")
    ax.set_xlabel(r"nondimensional time $(t/b)\sqrt{E/\rho}$", fontsize=11)
    ax.set_ylabel(r"nondimensional deflection $wEah/qb^3$", fontsize=11)
    ax.set_xlim(0, 9)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ex1_fig5_opensg_rm.png"), dpi=150,
                bbox_inches="tight")
    ipk = int(np.argmax(np.abs(wbar)))
    ipa = int(np.argmax(np.abs(wa)))
    print("wrote ex1_fig5_opensg_rm.png; RM peak w_bar = %.4f at t_bar ="
          " %.3f | analytic peak %.4f at %.3f (%+.1f%%)"
          % (wbar[ipk], tbar[ipk], wa[ipa], ta[ipa],
             100 * (wbar[ipk] - wa[ipa]) / wa[ipa]))


if __name__ == "__main__":
    main()
