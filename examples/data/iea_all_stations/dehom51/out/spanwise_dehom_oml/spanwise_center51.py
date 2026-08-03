"""spanwise_center51.py -- 51-station spanwise chain at the CENTER (mid-surface) reference
ONLY (the adopted blade chain): homogenization %err vs VABS .K and the suction-crown
recovery (stress + total disp).  Writes the paper figures directly:
  dehom_homo_pcterr_51_center.png, span_center_stress.png, span_center_disp.png.
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
SHELLD = os.path.join(ROOT, "shell51", "1d_yaml")            # center / mid-surface set
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


eta, diag, S, U, VS, VU = [], [], [], [], [], []
for i in range(51):
    smp = os.path.join(VABS, "iea_s%02d.sg.SM" % i); up = os.path.join(VABS, "iea_s%02d.sg.U" % i)
    kp = os.path.join(VABS, "iea_s%02d.sg.K" % i)
    shp = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not all(os.path.exists(p) for p in (smp, up, kp, shp)):
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
        B = dehom_rm.build_rm_bundle(shp); C6 = np.asarray(B["Timo"]); FF = FF_ALL[i, 1:]
        de = 100.0 * (np.diag(C6) - np.diag(Kv)) / np.diag(Kv)
        s = np.asarray(dehom_rm.stress_at_points(B, pt[None], beam_force_vabs=FF,
                       frame="material", n_per_layer=4, flow_avg=True)["stress"])[0]
        w = np.asarray(dehom_rm.disp_at_points(B, pt[None], beam_force_vabs=FF))[0]
        u_g, C = beam_kin(BD_OUT, i + 1); r3 = np.array([0.0, pt[0], pt[1]])
        eta.append(i / 50.0); diag.append(de); S.append([s[SVOIGT[k]] for k in BE])
        U.append(u_g + C @ (w + r3) - r3); VS.append(Vs.tolist()); VU.append(Vu.tolist())
        print("s%02d ok  GA3 %+6.2f  u3 %.3f" % (i, de[2], U[-1][2]), flush=True)
    except Exception as e:
        print("s%02d FAIL %s" % (i, str(e)[:60]), flush=True)

eta = np.array(eta); diag = np.array(diag); S = np.array(S); U = np.array(U)
VS = np.array(VS); VU = np.array(VU)
np.savez(os.path.join(HERE, "spanwise_center51.npz"), eta=eta, diag=diag, S=S, U=U, VS=VS, VU=VU)
print("\ncenter homo mean |%%err|: %s" % "  ".join(
    "%s %.2f" % (["EA", "GA2", "GA3", "GJ", "EI2", "EI3"][k], np.nanmean(np.abs(diag[:, k]))) for k in range(6)))

LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
TIT = ["extension $EA$", "transv. shear $GA_2$", "transv. shear $GA_3$",
       "torsion $GJ$", "flap bending $EI_2$", "edge bending $EI_3$"]
col = plt.cm.rainbow(np.linspace(0, 1, 6))
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14})
fig, axs = plt.subplots(3, 2, figsize=(12, 13.5))
for k in range(6):
    ax = axs.flat[k]
    ax.axhspan(-5, 5, color="0.9", zorder=0); ax.axhline(0, color="0.6", lw=1.2, ls=":")
    ax.plot(eta, diag[:, k], "-o", color=col[k], mec="k", mew=0.5, ms=8, lw=2.2)
    mx = np.nanmax(np.abs(diag[:, k])); ax.set_ylim(-max(6, 1.2 * mx), max(6, 1.2 * mx))
    ax.set_xlabel(r"span $r/R$"); ax.set_ylabel(r"$%s$ RM vs VABS  [\%%]" % LBL[k])
    ax.set_title(TIT[k], fontsize=15); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "dehom_homo_pcterr_51_center.png"), dpi=150); plt.close(fig)

VABSC = "#1f77b4"; RMC = "#ff7f0e"
SIN = [("11", 0), ("12", 1), ("22", 3)]; SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
d0 = np.abs(S[:, 0] - VS[:, 0]); keep = d0 <= 8.0 * np.median(d0) + 1e-12
fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
for ax, (k, idx), lab in zip(axs, SIN, SLAB):
    ax.plot(eta[keep], VS[keep, idx] / 1e6, "-o", color=VABSC, ms=6.5, lw=2.2, label="VABS")
    ax.plot(eta[keep], S[keep, idx] / 1e6, "--s", color=RMC, ms=6.5, mfc="none", mew=1.8,
            lw=2.0, label="RM shell")
    ax.set_xlabel(r"span  $r/R$"); ax.set_ylabel("%s   [MPa]" % lab); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(HERE, "span_center_stress.png"), dpi=150); plt.close(fig)

ULAB = [r"$u_1$ (out-of-plane warping)", r"$u_2$ (edgewise)", r"$u_3$ (flapwise)"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for k, (ax, lab) in enumerate(zip(axs, ULAB)):
    ax.plot(eta, VU[:, k], "-o", color=VABSC, ms=6.5, lw=2.2, label="VABS")
    ax.plot(eta, U[:, k], "--s", color=RMC, ms=6.5, mfc="none", mew=1.8, lw=2.0, label="RM shell")
    ax.set_xlabel(r"span  $r/R$"); ax.set_ylabel("%s   [m]" % lab); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(HERE, "span_center_disp.png"), dpi=150); plt.close(fig)
print("wrote center homo + span stress/disp")
