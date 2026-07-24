"""IEA-22 r/R=0.2 cross-section: RM homogenization + two-step dehomogenization vs VABS.

Standalone tutorial script -- runs entirely from data committed to this repository
(examples/data/iea_all_stations/...).  Clone the repo and run:

    python docs/tutorials/iea_r020_homo_dehom.py

It (1) homogenizes the IEA-22 r/R=0.2 blade cross-section with the Reissner-Mindlin (RM)
ring and compares the Timoshenko 6x6 against VABS, then (2) dehomogenizes -- recovers the
pointwise 3-D stress and warping displacement -- along THREE paths and overlays VABS:
    (a) circumferential (around the section, on the mid-surface contour),
    (b) LP spar-cap through-thickness (OML -> IML),
    (c) a connected cap -> T-junction -> web path (the C0 continuity demonstration).
All at the CENTER (mid-surface) reference.  Prints the numbers and saves the figures as
PNGs next to this file.
"""
import os
import sys
import time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ------------------------------------------------------------------ repo + imports
def _find_repo_root(d=None):
    d = os.path.abspath(d or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, "examples", "data")) and \
           os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("run this from inside the OpenSG-TW repo")
        d = p


CC = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
XSEC = os.path.join(CC, "examples", "TW-paper", "xsec_paper")
MITC = os.path.join(CC, "mitc_rm_segment")
for q in (CC, XSEC, MITC):
    if q not in sys.path:
        sys.path.insert(0, q)

import jax
jax.config.update("jax_enable_x64", True)

from xsec_5v6_master import load_ring, ring_6dof, LBL   # RM ring homogenization (Timoshenko 6x6)
import dehom_rm                                         # RM two-step dehomogenization

DATA = os.path.join(CC, "examples", "data", "iea_all_stations")
SHELL = os.path.join(DATA, "shell51", "1d_yaml", "iea_s10_shell.yaml")           # center-ref 1-D SG
KF = os.path.join(DATA, "dehom51", "out", "VABS_iea51", "iea_s10.sg.K")          # VABS 6x6 benchmark
FFF = os.path.join(DATA, "dehom51", "beamdyn", "ff51_rmc_reform.dat")            # beam forces (51 stations)
VBD = os.path.join(DATA, "dehom51", "out", "dehom_vabs")                         # VABS dehom overlays
JDAT = os.path.join(DATA, "dehom51", "out", "cpb_r020_msgrm", "data",
                    "junction_polyline_mid.dat")                                 # paper junction anchor
BD = os.path.join(DATA, "dehom51", "out", "VABS_iea51", "iea51vabs_bd_driver.out")  # BeamDyn kinematics
HERE = os.path.dirname(os.path.abspath(__file__))

VABSC = "#1f77b4"        # VABS   -> blue, solid, bold (the benchmark)
RMC = "#ff7f0e"          # OpenSG -> orange, dashed
# RM stress is Voigt [S11,S22,S33,S23,S13,S12]; map each VABS .out row to that index:
RM_OF = {"s_11": 0, "s_22": 1, "s_33": 2, "s_23": 3, "s_13": 4, "s_12": 5}


# ------------------------------------------------------------------ beam kinematics
# The dehom warping w (u1,u2,u3) is the *fluctuating* part; the physical local displacement
# recovered by VABS in the .out (u_1,u_2,u_3 ~ hundreds of mm) is the TOTAL:  u = u_g + C (w + r) - r,
# where (u_g, C) are the beam-node translation / rotation from the 1-D BeamDyn solve (same load path
# that produced FF).  We read them for node 11 (== station r/R=0.2) and reconstruct the RM total
# displacement so it is directly comparable to the VABS .out (mirrors cpb_r020_final.total_disp).
def _beam_kinematics(path, node):
    Lf = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(Lf):
        if l.strip().startswith("Time"):
            h = l.split()
            r = np.array([rr.split() for rr in Lf[i + 2:]], float)[-1]     # last time row
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")])
            RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]])                          # BeamDyn -> section axes
            t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
            return u_g, C
    raise ValueError("no BeamDyn time header in " + path)


_UG, _CBK = _beam_kinematics(BD, 11)


def total_disp_mm(w_mm, xy):
    """RM warping w (mm) at section points xy=(y2,y3) -> total local displacement (mm)."""
    r3 = np.column_stack([np.zeros(len(xy)), xy[:, 0], xy[:, 1]])
    return ((_CBK @ (w_mm / 1e3 + r3).T).T + _UG - r3) * 1e3


# ================================================================== 1. HOMOGENIZATION
def load_vabs_timo(path):
    """Parse the 6x6 Timoshenko stiffness block from a VABS .K file (section axes,
    order [EA, GA2, GA3, GJ, EI2, EI3])."""
    L = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i + 1:]:
        p = ln.split()
        ok = len(p) == 6
        try:
            [float(x) for x in p]
        except ValueError:
            ok = False
        if ok:
            rows.append([float(x) for x in p])
        if len(rows) == 6:
            break
    return np.array(rows)


def homogenize():
    t0 = time.perf_counter()
    C6 = np.asarray(ring_6dof(load_ring(SHELL)))   # center_ref=True (mid-surface) by default
    dt = time.perf_counter() - t0
    K = load_vabs_timo(KF)
    print("=" * 64)
    print("1. HOMOGENIZATION -- RM ring Timoshenko 6x6 vs VABS .K  (%.2f s)" % dt)
    print("=" * 64)
    print("%-5s %14s %14s %8s" % ("term", "VABS .K", "RM ring", "%err"))
    for i in range(6):
        v, r = K[i, i], C6[i, i]
        print("%-5s %14.4e %14.4e %+7.2f" % (LBL[i], v, r, 100 * (r - v) / v))
    fro = np.linalg.norm(C6 - K) / np.linalg.norm(K) * 100
    print("full-6x6 Frobenius error = %.2f%%   (all diagonal terms within ~2.7%%)" % fro)
    return C6, K


# ================================================================== 2. DEHOM helpers
def read_out(path):
    """VABS dehom .out (row-keyed): returns {row_label: values}."""
    d = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def sample_path(B, P, FF):
    """RM two-step recovery at query points P (N,2): material-frame stress (MPa) and
    TOTAL local displacement (mm) = beam rigid motion + recovered warping."""
    S = np.asarray(dehom_rm.stress_at_points(B, P, beam_force_vabs=FF, frame="material")["stress"]) / 1e6
    w = np.asarray(dehom_rm.disp_at_points(B, P, beam_force_vabs=FF)) * 1e3    # warping (mm)
    U = total_disp_mm(w, P)                                                    # total (mm)
    return S, U


def rel_pct(rm, va):
    return 100.0 * np.linalg.norm(rm - va) / (np.linalg.norm(va) + 1e-30)


# ---- plotting: 6-panel (row1 stress s11,s12,s22 ; row2 disp u1,u2,u3) ----
SCOMP = ["s_11", "s_12", "s_22"]
SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
ULAB = [r"$u_1$", r"$u_2$", r"$u_3$"]


def _plain(ax):
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(alpha=0.3, ls=":")


def plot_overlay(xs, S, U, vb, keep, xlabel, fname):
    """RM (orange dashed) vs VABS (blue solid, bold) along a path.  ``keep`` masks
    out projection-outlier points (junction/thick-region) from every panel.  Top row =
    material-frame stress (MPa); bottom row = total local displacement (mm)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for k in range(3):
        a = ax[0, k]
        va = vb[SCOMP[k]] / 1e6
        a.plot(xs[keep], va[keep], "-o", color=VABSC, ms=3.5, lw=2.2, label="VABS")
        a.plot(xs[keep], S[keep, RM_OF[SCOMP[k]]], "--s", color=RMC, ms=3.5, mfc="none",
               mew=1.2, lw=1.4, label="OpenSG-RM")
        a.set_ylabel("%s  [MPa]" % SLAB[k]); a.set_xlabel(xlabel); _plain(a); a.legend(fontsize=8)
    for k in range(3):
        a = ax[1, k]
        vu = vb["u_%d" % (k + 1)] * 1e3                      # VABS total local displacement (mm)
        a.plot(xs[keep], vu[keep], "-o", color=VABSC, ms=3.5, lw=2.2, label="VABS")
        a.plot(xs[keep], U[keep, k], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2, lw=1.4,
               label="OpenSG-RM")
        a.set_ylabel("%s  [mm]" % ULAB[k]); a.set_xlabel(xlabel); _plain(a); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=150)
    plt.close(fig)
    print("  wrote", fname)


def dehom_line(B, FF, stem, xlabel, fname, mask_outliers):
    vb = read_out(os.path.join(VBD, stem + ".out"))
    P = np.column_stack([vb["y2"], vb["y3"]])
    t0 = time.perf_counter()
    S, U = sample_path(B, P, FF)
    dt = time.perf_counter() - t0
    xs = vb["non_dim_path"]
    if mask_outliers:                       # circumferential path crosses webs/junctions:
        resid = np.abs(S[:, 0] - vb["s_11"] / 1e6)     # a handful of contour points project
        keep = resid <= 8.0 * np.median(resid)         # onto a thick wall -> spike; drop them
    else:
        keep = np.ones(len(P), bool)
    nhid = int((~keep).sum())
    print("  %-34s  %d pts, %d hidden, dehom %.2f s" % (stem, len(P), nhid, dt))
    print("     stress:  " + "   ".join("%s %.1f%%" % (c, rel_pct(S[keep, RM_OF[c]], (vb[c] / 1e6)[keep]))
                                        for c in ("s_11", "s_12", "s_22")))
    print("     disp  :  " + "   ".join("u%d %.2f%%" % (k + 1, rel_pct(U[keep, k], vb["u_%d" % (k + 1)][keep] * 1e3))
                                        for k in range(3)))
    plot_overlay(xs, S, U, vb, keep, xlabel, fname)
    return S, U, vb, keep


# ================================================================== 3c. JUNCTION path
def _web_mask(corners, rc, deg, adj, Lel):
    """Classify ring elements as shear-web (long, near-vertical chain between two
    junctions) -- the same rule the paper (cpb_r020_final.py) uses."""
    junc = set(np.where(deg >= 3)[0])
    is_web = np.zeros(len(rc), bool)
    seen = set()
    for j in junc:
        for (nxt, e0) in adj[j]:
            if e0 in seen:
                continue
            chain, prev, cur = [e0], j, nxt
            seen.add(e0)
            while cur not in junc and deg[cur] == 2:
                (n1, e1), (n2, e2) = adj[cur][0], adj[cur][1]
                nn, ee = (n1, e1) if n1 != prev else (n2, e2)
                if ee in seen:
                    break
                chain.append(ee)
                seen.add(ee)
                prev, cur = cur, nn
            if cur in junc:
                arc = sum(Lel[c] for c in chain)
                cv = corners[cur] - corners[j]
                ch = float(np.linalg.norm(cv))
                if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                    is_web[chain] = True
    return is_web, junc


def junction_path(B, ns=6, nw=2):
    """Build a connected CONTOUR polyline: cap skin -> spar-cap/web T-junction -> web,
    all on the mid-surface (depth z=0).  Anchored to the paper's junction location
    (junction_polyline_mid.dat) when that file is present."""
    corners = np.asarray(B["corners"])
    rc = np.asarray(B["red_cells"])
    layups = B["layup_per_elem"]
    n_nd = int(rc.max()) + 1
    deg = np.zeros(n_nd, int)
    adj = [[] for _ in range(n_nd)]
    for e, (a, b) in enumerate(rc):
        deg[a] += 1
        deg[b] += 1
        adj[a].append((int(b), e))
        adj[b].append((int(a), e))
    Lel = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
    is_web, junc = _web_mask(corners, rc, deg, adj, Lel)
    jl = [nd for nd in junc if corners[nd, 1] > 0]
    if os.path.exists(JDAT):
        anchor = np.loadtxt(JDAT)[0]                      # paper's junction, upper spar-cap/web T
        ndj = jl[int(np.argmin([np.hypot(*(corners[nd] - anchor)) for nd in jl]))]
    else:
        ndj = jl[int(np.argmin([abs(corners[nd, 0] + 0.044) for nd in jl]))]

    def walk(want_web, n):
        seq, cur, prev = [ndj], ndj, None
        for _ in range(n):
            cand = [(nb, e) for (nb, e) in adj[cur]
                    if nb != prev and is_web[e] == want_web and nb not in junc]
            if not cand:
                break
            nb, e = cand[0]
            seq.append(nb)
            prev, cur = cur, nb
        return seq

    nodes = list(reversed(walk(False, ns))) + walk(True, nw)[1:]
    pc = corners[nodes]
    seg = []
    for a, b in zip(pc[:-1], pc[1:]):
        m = max(3, int(np.hypot(*(b - a)) / 0.004))
        seg.append(np.linspace(a, b, m, endpoint=False))
    seg.append(pc[-1][None])
    PL = np.vstack(seg)
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(PL[:, 0]), np.diff(PL[:, 1])))] * 1e3
    junc_arc = float(np.hypot(*(corners[ndj] - pc[0]))) * 1e3
    return PL, arc, junc_arc


def dehom_junction(B, FF, fname):
    PL, arc, jarc = junction_path(B)
    S, U = sample_path(B, PL, FF)
    dmax = np.abs(np.diff(U, axis=0)).max(0)
    print("  junction path: %d samples, arc 0..%.0f mm, T-junction at %.0f mm" % (len(PL), arc[-1], jarc))
    print("     displacement continuity: max adjacent |du1|=%.4f |du2|=%.4f |du3|=%.4f mm"
          % (dmax[0], dmax[1], dmax[2]))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for k in range(3):
        a = ax[0, k]
        a.plot(arc, S[:, RM_OF[SCOMP[k]]], "-", color=RMC, lw=1.8)
        a.axvline(jarc, color="0.5", ls=":", lw=1.2)
        a.set_ylabel("%s  [MPa]" % SLAB[k]); a.set_xlabel("arc along path  [mm]"); _plain(a)
    for k in range(3):
        a = ax[1, k]
        a.plot(arc, U[:, k], "-", color=RMC, lw=1.8)
        a.axvline(jarc, color="0.5", ls=":", lw=1.2)
        a.set_ylabel("%s  [mm]" % ULAB[k]); a.set_xlabel("arc along path  [mm]"); _plain(a)
    # mark the cap / web T-junction (the dotted line) on every panel
    for a in ax.flat:
        a.text(jarc, 0.03, " cap | web\n T-junction", transform=a.get_xaxis_transform(),
               ha="left", va="bottom", fontsize=7.5, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=150)
    plt.close(fig)
    print("  wrote", fname)


# ================================================================== main
def main():
    homogenize()
    FF = np.loadtxt(FFF)[10, 1:]     # r/R=0.2 beam force resultant, VABS order [F1,F2,F3,M1,M2,M3]
    print("\n" + "=" * 64)
    print("2. DEHOMOGENIZATION  (FF = %s)" % np.array2string(FF, precision=3))
    print("=" * 64)
    t0 = time.perf_counter()
    B = dehom_rm.build_rm_bundle(SHELL, ref="center")
    print("  built RM dehom bundle in %.2f s" % (time.perf_counter() - t0))
    print("\n(a) circumferential path")
    dehom_line(B, FF, "iea_s10.circumferential", "non-dimensional path coordinate",
               "iea_r020_dehom_circ.png", mask_outliers=True)
    print("\n(b) LP spar-cap through-thickness path (OML -> IML)")
    dehom_line(B, FF, "iea_s10.lp_sparcap_left_thickness", "non-dimensional path (OML -> IML)",
               "iea_r020_dehom_cap.png", mask_outliers=False)
    print("\n(c) cap -> T-junction -> web continuity path")
    dehom_junction(B, FF, "iea_r020_dehom_junction.png")
    print("\nDONE")


if __name__ == "__main__":
    main()
