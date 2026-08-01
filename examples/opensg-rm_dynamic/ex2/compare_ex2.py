"""compare_ex2.py -- Nayak Example 2: OpenSG-RM (Abaqus S4 + MSG ABDG)
against the KHDEIR-REDDY 1989 EXACT ANALYTICAL SOLUTION (Nayak's ref [11],
PDF in the OneDrive literature folder).

THE ANALYTICAL REFERENCE: Khdeir & Reddy solve exactly this plate --
(0/90/0), a = b = 5h, h = 0.1524 m, E1/E2 = 25 set, q0 = 68.9476 MPa
double-sine load, t1 = 6 ms, gamma = 330 1/s -- with the higher-order
theory, EXACTLY in space (single Navier mode) and EXACTLY in time (state
space / modal superposition; "Both procedures give the same result").
Their paper has NO tables (figures only, deflection in INCHES -- hence
Nayak's w/0.0254 m scaling), so the reference curve here is the same
closed-form solution evaluated by ../ex5/reddy_hsdt_navier.py with
set_case("ex2") -- which is pinned to Nayak's converged Table-3 FE values
within 2-7 % (his dt = 40 us bias).

COMPARED QUANTITIES (both pulses: step cut off at t1, blast e^(-330 t)):
  * center deflection w/0.0254 m (their Fig.-1 quantity);
  * the top-surface normal stress sigma_x(a/2, b/2, h/2)/q0 (their
    sigma1_bar of Fig. 2), recovered from the Abaqus SF/SM patch prints
    through the SG chain -- the analytic twin from the TSDT constitutive
    stresses at ITS deflection history.

Outputs: ex2_w_history_{step,blast}.png, ex2_sx_history_{step,blast}.png
(each individual, own legend: Khdeir-Reddy exact = dashed black,
OpenSG-RM/Abaqus = orange markers) + ex2_results.dat (peaks + the values
at Nayak's anchor times 1.6/3.2/4.8/6.4/8.0 ms vs his Table-3 numbers).

Run:  python examples/opensg-rm_dynamic/ex2/compare_ex2.py
"""
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "yu2003"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ex5"))

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from recover_6p2 import read_elprint_tables                     # noqa: E402
import reddy_hsdt_navier as rd                                  # noqa: E402

H = 0.1524
A = 5.0 * H
NX = 20
Q0 = 68.9476e6
T1 = 0.006
CBLAST = 330.0
P = np.pi / A
XC = (A / 2, A / 2)
# Nayak Table 3 (9-node, 4x4 mesh, dt = 40 us -- his converged FE) at
# t = 1.6/3.2/4.8/6.4/8.0 ms, step pulse: w/0.0254 and sigma_x_bar
T_ANCH = np.array([1.6e-3, 3.2e-3, 4.8e-3, 6.4e-3, 8.0e-3])
NAYAK_W = [0.1870, 0.6180, 0.9983, 0.4548, -0.2191]
NAYAK_SX = [5.9784, 19.8407, 32.2827, 14.7361, -6.8995]

inp = read_plate_sg_yaml(os.path.join(HERE, "ex2_sg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
S6 = np.linalg.inv(np.asarray(r["A6"]))


def step_times(dat_path):
    ts = []
    with open(dat_path, errors="replace") as f:
        for ln in f:
            m = re.match(r"\s*STEP TIME COMPLETED\s+([0-9.E+-]+)", ln)
            if m:
                ts.append(float(m.group(1)))
    return np.array(ts)


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
            if vals:
                rows.append(vals)
            active = False
    return np.array(rows)


def patch_fit(tables):
    """PATCHC SF/SM series fitted about the center: (n_inc, 8, 3)."""
    sf = None; xy = None
    for (es, labels), rows in tables.items():
        if es != "PATCHC":
            continue
        ofs = rows.shape[1] - len(labels)
        if "SF1" in labels:
            idx = [labels.index(k) + ofs for k in
                   ("SF1", "SF2", "SF3", "SF4", "SF5", "SM1", "SM2", "SM3")]
            sf = rows[:, idx].reshape(-1, 16, 8)
        if "COORD1" in labels:
            idx = [labels.index(k) + ofs for k in ("COORD1", "COORD2")]
            xy = rows[:16, idx]
    xi, eta = xy[:, 0] - XC[0], xy[:, 1] - XC[1]
    B = np.column_stack([np.ones(16), xi, eta, xi * eta, xi ** 2, eta ** 2])
    out = np.empty((sf.shape[0], 8, 3))
    for k in range(sf.shape[0]):
        C, *_ = np.linalg.lstsq(B, sf[k], rcond=None)
        out[k] = np.column_stack([C[0], C[1], C[2]])
    return out


def ft_of(kind, t):
    if kind == "step":
        return np.where(t <= T1, 1.0, 0.0)
    return np.exp(-CBLAST * t)


def rm_sx_top(fits, t_rm, kind):
    """The recovered sigma_x(a/2, b/2, h/2)/q0 history from the S4 patch
    resultants: E6 = S6 R6 + mode-consistent second gradients + the load
    ladder, evaluated one Gauss-free point below the top surface."""
    z_top = H / 2 - 1e-6
    z6 = np.zeros(6)
    out = np.empty(len(t_rm))
    for k in range(len(t_rm)):
        F = fits[k]
        E6 = S6 @ F[[0, 1, 2, 5, 6, 7], 0]
        dE1 = S6 @ F[[0, 1, 2, 5, 6, 7], 1]
        dE2 = S6 @ F[[0, 1, 2, 5, 6, 7], 2]
        qt6 = float(ft_of(kind, t_rm[k])) * Q0 * np.array(
            [1.0, 0, 0, -P * P, 0, -P * P])
        out[k] = msgrm_strain_at_depth(r, z_top, E6, dE1, dE2,
                                       -P * P * E6, z6, -P * P * E6,
                                       qt6=qt6)[1][0]
    return out / Q0


def one_plot(fname, curves, xlabel, ylabel, xlim=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for x, y, sty, lab in curves:
        ax.plot(x, y, label=lab, **sty)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# four processes -> four DISTINCT markers, staggered so they never overlap;
# histories are windowed to 0-8 ms (the KR-1989 figure window) for clarity
STY_SOL = dict(ls="--", marker="o", color="k", lw=1.6, ms=5, mfc="none",
               mew=1.2, markevery=(0, 14))                # 3-D benchmark
STY_RM = dict(ls="-", marker="s", color="#ff7f0e", lw=1.3, ms=5,
              mfc="none", mew=1.2, markevery=(4, 12))     # OpenSG-RM
STY_KR = dict(ls="-", marker="v", color="#4878a8", lw=1.2, ms=5,
              mfc="none", mew=1.1, markevery=(30, 60))    # exact HSDT
STY_FS = dict(ls="-.", marker="^", color="#2ca02c", lw=1.2, ms=5,
              mfc="none", mew=1.1, markevery=(8, 12))     # Abaqus FSDT
NZP = 6                        # solid elements per ply (18 through h)


def solid_w(dat):
    """(t, w) of the solid center node, sign-flipped (P2 loads -z while
    the shells load +z), times parsed (automatic increments)."""
    t_all = step_times(dat)
    uc = node_history(dat, "NCEN3D")
    return t_all[len(t_all) - len(uc):], -uc[:, 2]


def solid_sx_top(dat):
    """(t_dumps, sx_bar) at the top-layer element centroid of the center
    column (0-deg ply: local frame = global), sign-flipped."""
    t_all = step_times(dat)
    tables = read_elprint_tables(dat)
    nzt = 3 * NZP
    for (es, labels), rows in tables.items():
        if es == "COLC" and "S11" in labels:
            ofs = rows.shape[1] - len(labels)
            v = rows[:, ofs + labels.index("S11")].reshape(-1, nzt, 4)
            td = list(t_all[9::10])
            if v.shape[0] == len(td) + 1:      # Abaqus always dumps the
                td.append(t_all[-1])           # LAST increment too
            nd = min(len(td), v.shape[0])
            return np.array(td[:nd]), -v[:nd, -1, :].mean(axis=1) / Q0
    return None, None


def main():
    rd.set_case("ex2")
    Kt, Mt = rd.navier_KM("tsdt")
    tg = np.arange(0.0, 0.02 + 1e-12, 1.0e-5)
    z_ctr = H / 2 - H / (2 * 3 * NZP)      # solid top-element centroid z
    lines = ["Nayak Ex.2 / Khdeir-Reddy 1989 (0/90/0) a=5h -- FOUR-WAY:",
             "  benchmark = Abaqus 3-D solid (KR's 'exact' is exact only"
             " WITHIN the HSDT -- unlike Pagano it is NOT 3-D elasticity)",
             "  vs OpenSG-RM (Abaqus S4 + MSG ABDG), conventional Abaqus"
             " FSDT (S4 composite section), and the exact HSDT (KR 1989)"]
    for kind, pulse in (("step", "step_t1"), ("blast", "blast")):
        f_rm = os.path.join(HERE, "Abaqus_results", "ex2_RM_%s.dat" % kind)
        f_fs = os.path.join(HERE, "Abaqus_results", "ex2_FSDT_%s.dat" % kind)
        f_so = os.path.join(HERE, "Abaqus_results", "ex2_SOLID_%s.dat" % kind)
        if not os.path.isfile(f_rm):
            print("missing %s -- run the Abaqus job first" % f_rm)
            continue
        # ---- exact Khdeir-Reddy HSDT (closed-form modal, = their method)
        w_ex, lam, V, etas = rd.modal_response(Kt, Mt, pulse, tg, full=True)
        d_hist = V @ etas
        sx_ex = np.array([rd.tsdt_profiles(d_hist[:, k],
                                           [H / 2 - 1e-9])[0][0]
                          for k in range(len(tg))]) / Q0
        # ---- Abaqus OpenSG-RM -----------------------------------------
        t_all = step_times(f_rm)
        uc = node_history(f_rm, "NCEN")
        t_rm = t_all[len(t_all) - len(uc):]
        w_rm = uc[:, 2]
        fits = patch_fit(read_elprint_tables(f_rm))
        n = min(len(t_rm), len(w_rm), fits.shape[0])
        sx_rm = rm_sx_top(fits[:n], t_rm[:n], kind)
        # ---- Abaqus conventional FSDT ---------------------------------
        w_curves = [None, None]
        if os.path.isfile(f_fs):
            t_fs_all = step_times(f_fs)
            uf = node_history(f_fs, "NCEN")
            w_curves[0] = (t_fs_all[len(t_fs_all) - len(uf):], uf[:, 2])
        # ---- Abaqus 3-D solid (the benchmark) -------------------------
        if os.path.isfile(f_so):
            w_curves[1] = solid_w(f_so)
        # ---- figures --------------------------------------------------
        curves = []
        if w_curves[1] is not None:
            t_so, w_so = w_curves[1]
            curves.append((1e3 * np.concatenate([[0], t_so]),
                           np.concatenate([[0], w_so]) / 0.0254, STY_SOL,
                           "Abaqus 3-D solid\n(benchmark)"))
        curves.append((1e3 * np.concatenate([[0], t_rm[:n]]),
                       np.concatenate([[0], w_rm[:n]]) / 0.0254, STY_RM,
                       "OpenSG-RM\n(Abaqus S4 + MSG ABDG)"))
        if w_curves[0] is not None:
            t_fs, w_fs = w_curves[0]
            curves.append((1e3 * np.concatenate([[0], t_fs]),
                           np.concatenate([[0], w_fs]) / 0.0254, STY_FS,
                           "Abaqus FSDT\n(composite S4)"))
        curves.append((1e3 * tg, w_ex / 0.0254, STY_KR,
                       "Khdeir-Reddy 1989\nexact HSDT"))
        one_plot("ex2_w_history_%s.png" % kind, curves, "time [ms]",
                 r"center deflection $w/0.0254$ m", xlim=(0.0, 8.0))
        sx_curves = []
        if os.path.isfile(f_so):
            t_sd, sx_so = solid_sx_top(f_so)
            if t_sd is not None:
                sx_curves.append((1e3 * t_sd, sx_so, STY_SOL,
                                  "Abaqus 3-D solid\n(top element centroid)"))
        sx_curves += [(1e3 * t_rm[:n], sx_rm, STY_RM,
                       "OpenSG-RM recovery\n(z = h/2)"),
                      (1e3 * tg, sx_ex, STY_KR,
                       "Khdeir-Reddy exact\nHSDT (z = h/2)")]
        one_plot("ex2_sx_history_%s.png" % kind, sx_curves, "time [ms]",
                 r"$\bar\sigma_x = \sigma_x(a/2,b/2,z)/q_0$",
                 xlim=(0.0, 8.0))
        # ---- numbers --------------------------------------------------
        lines += ["", "%s pulse -- peak |w| [in = w/0.0254 m]:" % kind]

        def pk(t, w):
            i = int(np.argmax(np.abs(w)))
            return w[i] / 0.0254, 1e3 * t[i]

        entries = []
        if w_curves[1] is not None:
            entries.append(("Abaqus 3-D solid (benchmark)",
                            *pk(*w_curves[1])))
        entries.append(("OpenSG-RM (Abaqus S4+ABDG)", *pk(t_rm[:n],
                                                          w_rm[:n])))
        if w_curves[0] is not None:
            entries.append(("Abaqus FSDT (composite S4)", *pk(*w_curves[0])))
        entries.append(("Khdeir-Reddy exact HSDT", *pk(tg, w_ex)))
        wref = entries[0][1] if w_curves[1] is not None else None
        for nam, wpk, tpk in entries:
            err = ("  %+7.2f %% vs solid" % (100 * (wpk - wref) / wref)
                   if wref is not None else "")
            lines.append("  %-30s %8.4f in @ %6.2f ms%s"
                         % (nam, wpk, tpk, err))
        if kind == "step":
            wi = np.interp(T_ANCH, t_rm[:n], w_rm[:n]) / 0.0254
            we = np.interp(T_ANCH, tg, w_ex) / 0.0254
            si = np.interp(T_ANCH, t_rm[:n], sx_rm)
            se = np.interp(T_ANCH, tg, sx_ex)
            lines += ["  anchor times [ms]:      "
                      + " ".join("%8.1f" % (1e3 * t) for t in T_ANCH),
                      "  w/0.0254 KR-exact:      "
                      + " ".join("%8.4f" % v for v in we),
                      "  w/0.0254 Nayak-P9-FE:   "
                      + " ".join("%8.4f" % v for v in NAYAK_W),
                      "  w/0.0254 OpenSG-RM:     "
                      + " ".join("%8.4f" % v for v in wi)]
            if w_curves[1] is not None:
                ws = np.interp(T_ANCH, *w_curves[1]) / 0.0254
                lines.append("  w/0.0254 Abaqus solid:  "
                             + " ".join("%8.4f" % v for v in ws))
            if w_curves[0] is not None:
                wf = np.interp(T_ANCH, *w_curves[0]) / 0.0254
                lines.append("  w/0.0254 Abaqus FSDT:   "
                             + " ".join("%8.4f" % v for v in wf))
            lines += ["  sx_bar   KR-exact:      "
                      + " ".join("%8.3f" % v for v in se),
                      "  sx_bar   Nayak-P9-FE:   "
                      + " ".join("%8.3f" % v for v in NAYAK_SX),
                      "  sx_bar   OpenSG-RM:     "
                      + " ".join("%8.3f" % v for v in si)]
        print("wrote ex2_*_%s figures" % kind)
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(HERE, "ex2_results.dat"), "w") as f:
        f.write(out + "\n")
    rd.set_case("ex5")


if __name__ == "__main__":
    main()
