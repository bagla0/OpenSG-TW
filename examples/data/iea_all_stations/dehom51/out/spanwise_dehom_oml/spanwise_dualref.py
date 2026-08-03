"""spanwise_dualref.py -- 51-station spanwise comparison at BOTH references
(center / mid-surface AND OML), each vs VABS: homogenization %err (6 panels) and the
suction-crown recovery (stress + total disp).  center-ref is the accuracy chain, OML
the simple/convention chain; both overlaid so the reader sees center is slightly closer.

Outputs (this folder): dual_homo_pcterr_51.png, dual_span_stress.png, dual_span_disp.png,
and dual_spanwise_{center,oml}.out.
"""
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DEHOM = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(DEHOM, ".."))
VABS = os.path.join(DEHOM, "out", "VABS_iea51")
SHELL = {"center": os.path.join(ROOT, "shell51", "1d_yaml"),
         "oml": os.path.join(ROOT, "shell51", "1d_yaml_oml51")}
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
import jax

jax.config.update("jax_enable_x64", True)
import dehom_rm

FF_ALL = np.loadtxt(os.path.join(DEHOM, "beamdyn", "ff51_rmc_reform.dat"))
BD_OUT = os.path.join(DEHOM, "beamdyn", "iea51rmc_bd_driver.out")
BE = ("11", "12", "13", "22", "23", "33")
SVOIGT = {"11": 0, "12": 5, "13": 4, "22": 1, "23": 3, "33": 2}


def beam_kin(path, node):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); row = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: row[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            return u_g, np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
    raise ValueError("no header")


def block6(L, key):
    for i, l in enumerate(L):
        if key.lower() in l.lower():
            rows, j = [], i + 1
            while len(rows) < 6 and j < len(L):
                try:
                    v = [float(x) for x in L[j].split()]
                    if len(v) >= 6:
                        rows.append(v[:6])
                except ValueError:
                    pass
                j += 1
            return np.array(rows)
    return None


REFS = ("center", "oml")
eta = []
diag = {r: [] for r in REFS}
S = {r: [] for r in REFS}
U = {r: [] for r in REFS}
VS, VU = [], []
for i in range(51):
    smp = os.path.join(VABS, "iea_s%02d.sg.SM" % i); up = os.path.join(VABS, "iea_s%02d.sg.U" % i)
    kp = os.path.join(VABS, "iea_s%02d.sg.K" % i)
    if not all(os.path.exists(p) for p in (smp, up, kp)):
        continue
    ok = all(os.path.exists(os.path.join(SHELL[r], "iea_s%02d_shell.yaml" % i)) for r in REFS)
    if not ok:
        continue
    try:
        SM = np.loadtxt(smp, skiprows=2); Uu = np.loadtxt(up)
        Kv = block6([l for l in open(kp).read().splitlines()], "Timoshenko Stiffness Matrix")
        band = 0.10; sel = np.abs(SM[:, 0]) < band
        if sel.sum() < 3:
            sel = np.abs(SM[:, 0]) < 0.30
        cand = np.where(sel)[0]; itop = int(cand[np.argmax(SM[cand, 1])]); pt = SM[itop, :2]
        Vs = SM[itop, 2:8]
        uxy = Uu[:, 1:3]; uv = Uu[:, 3:6]
        dU, iU = cKDTree(uxy).query(pt[None], k=4)
        wv = 1.0 / (dU + 1e-8 * (dU.sum(1, keepdims=True) + 1e-30)); wv /= wv.sum(1, keepdims=True)
        Vu = np.einsum("pk,pkj->pj", wv, uv[iU])[0]
        u_g, C = beam_kin(BD_OUT, i + 1); r3 = np.array([0.0, pt[0], pt[1]]); FF = FF_ALL[i, 1:]
        rowS, rowU, rowD = {}, {}, {}
        good = True
        for r in REFS:
            B = dehom_rm.build_rm_bundle(os.path.join(SHELL[r], "iea_s%02d_shell.yaml" % i))
            C6 = np.asarray(B["Timo"])
            rowD[r] = 100.0 * (np.diag(C6) - np.diag(Kv)) / np.diag(Kv)
            s = np.asarray(dehom_rm.stress_at_points(B, pt[None], beam_force_vabs=FF,
                           frame="material", n_per_layer=4, flow_avg=True)["stress"])[0]
            w = np.asarray(dehom_rm.disp_at_points(B, pt[None], beam_force_vabs=FF))[0]
            rowS[r] = [s[SVOIGT[k]] for k in BE]
            rowU[r] = u_g + C @ (w + r3) - r3
        eta.append(i / 50.0); VS.append(Vs.tolist()); VU.append(Vu.tolist())
        for r in REFS:
            diag[r].append(rowD[r]); S[r].append(rowS[r]); U[r].append(rowU[r])
        print("s%02d  GA3 center %+6.2f / OML %+6.2f  u3 %.3f" % (i, rowD["center"][2], rowD["oml"][2],
              rowU["center"][2]), flush=True)
    except Exception as e:
        print("s%02d FAIL %s" % (i, str(e)[:60]), flush=True)

eta = np.array(eta)
diag = {r: np.array(diag[r]) for r in REFS}
S = {r: np.array(S[r]) for r in REFS}; U = {r: np.array(U[r]) for r in REFS}
VS = np.array(VS); VU = np.array(VU)
print("\n%d stations" % len(eta))
for r in REFS:
    print("  %-7s homo mean |%%err|: %s" % (r, "  ".join(
        "%s %.2f" % (["EA", "GA2", "GA3", "GJ", "EI2", "EI3"][k], np.nanmean(np.abs(diag[r][:, k])))
        for k in range(6))))

# ---- homogenization %err, both refs ----
LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
TIT = ["extension $EA$", "transv. shear $GA_2$", "transv. shear $GA_3$",
       "torsion $GJ$", "flap bending $EI_2$", "edge bending $EI_3$"]
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12})
fig, axs = plt.subplots(3, 2, figsize=(12, 13.0))
for k in range(6):
    ax = axs.flat[k]
    ax.axhspan(-5, 5, color="0.9", zorder=0); ax.axhline(0, color="0.6", lw=1.0, ls=":")
    ax.plot(eta, diag["center"][:, k], "--s", color="#ff7f0e", mec="k", mew=0.4, ms=6,
            lw=1.8, label="center", mfc="none")
    ax.plot(eta, diag["oml"][:, k], ":^", color="#2ca02c", mec="k", mew=0.4, ms=6,
            lw=1.8, label="OML", mfc="none")
    mx = max(np.nanmax(np.abs(diag["center"][:, k])), np.nanmax(np.abs(diag["oml"][:, k])))
    ax.set_ylim(-max(6, 1.2 * mx), max(6, 1.2 * mx))
    ax.set_xlabel(r"span $r/R$"); ax.set_ylabel(r"$%s$ vs VABS  [\%%]" % LBL[k])
    ax.set_title(TIT[k], fontsize=14); ax.grid(alpha=0.25)
    if k == 0:
        ax.legend(loc="best")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "dual_homo_pcterr_51.png"), dpi=150); plt.close(fig)

# ---- spanwise stress + disp, both refs ----
VABSC = "#1f77b4"; CENC = "#ff7f0e"; OMLC = "#2ca02c"
SIN = [("11", 0), ("12", 1), ("22", 3)]; SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
d0 = np.abs(np.array(S["center"])[:, 0] - VS[:, 0]); keep = d0 <= 8.0 * np.median(d0) + 1e-12
fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
for ax, (k, idx), lab in zip(axs, SIN, SLAB):
    ax.plot(eta[keep], VS[keep, idx] / 1e6, "-o", color=VABSC, ms=5.5, lw=2.0, label="VABS")
    ax.plot(eta[keep], S["center"][keep, idx] / 1e6, "--s", color=CENC, ms=5.5, mfc="none",
            mew=1.4, lw=1.7, label="RM (center)")
    ax.plot(eta[keep], S["oml"][keep, idx] / 1e6, ":^", color=OMLC, ms=5.5, mfc="none",
            mew=1.4, lw=1.7, label="RM (OML)")
    ax.set_xlabel(r"span  $r/R$"); ax.set_ylabel("%s   [MPa]" % lab); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(HERE, "dual_span_stress.png"), dpi=150); plt.close(fig)

ULAB = [r"$u_1$ (out-of-plane warping)", r"$u_2$ (edgewise)", r"$u_3$ (flapwise)"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for k, (ax, lab) in enumerate(zip(axs, ULAB)):
    ax.plot(eta, VU[:, k], "-o", color=VABSC, ms=5.5, lw=2.0, label="VABS")
    ax.plot(eta, U["center"][:, k], "--s", color=CENC, ms=5.5, mfc="none", mew=1.4, lw=1.7, label="RM (center)")
    ax.plot(eta, U["oml"][:, k], ":^", color=OMLC, ms=5.5, mfc="none", mew=1.4, lw=1.7, label="RM (OML)")
    ax.set_xlabel(r"span  $r/R$"); ax.set_ylabel("%s   [m]" % lab); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(HERE, "dual_span_disp.png"), dpi=150); plt.close(fig)
print("wrote dual_homo_pcterr_51.png, dual_span_stress.png, dual_span_disp.png")
