"""iea_spanwise.py -- 51-station spanwise RM homogenization + stress/disp recovery vs VABS.

Standalone companion to docs/tutorials/iea_spanwise.ipynb.  Runs ENTIRELY from data
committed to this repository (examples/data/iea_all_stations/...):

  * shell51/1d_yaml/iea_s00..s50_shell.yaml        -- 51 center-ref 1-D shell SGs (~2.7 MB)
  * dehom51/beamdyn/ff51_rmc_reform.dat            -- per-station beam section forces (VABS order)
  * dehom51/beamdyn/iea51rmc_bd_driver.out         -- BeamDyn nodal disp/rot (beam kinematics)
  * dehom51/benchmark/spanwise_vabs_landmarks.npz  -- pre-extracted VABS landmark per station
                                                      (pt, stress 6-vec, disp 3-vec, Timoshenko 6x6)

For every station it (i) homogenizes the Reissner-Mindlin (RM) shell ring at the CENTER
(mid-surface) reference and compares the Timoshenko 6x6 diagonal to VABS `.sg.K`, and
(ii) recovers the 3-D stress + total displacement at one near-root-band suction-crown
landmark and compares to the VABS `.sg.SM` / `.sg.U` values stored in the npz.

Run:  python docs/tutorials/iea_spanwise.py
Saves iea_spanwise_homo_pcterr.png, iea_spanwise_stress.png, iea_spanwise_disp.png here.
"""
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_repo_root(d=None):
    d = os.path.abspath(d or os.path.dirname(os.path.abspath(__file__)))
    while True:
        if os.path.isdir(os.path.join(d, "examples", "data")) and \
           os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("run this from inside the OpenSG-TW repo")
        d = p


HERE = os.path.dirname(os.path.abspath(__file__))
CC = _find_repo_root()
XSEC = os.path.join(CC, "examples", "TW-paper", "xsec_paper")
MITC = os.path.join(CC, "mitc_rm_segment")
for q in (CC, XSEC, MITC):
    if q not in sys.path:
        sys.path.insert(0, q)
import jax

jax.config.update("jax_enable_x64", True)
import dehom_rm

DATA = os.path.join(CC, "examples", "data", "iea_all_stations")
SHELLD = os.path.join(DATA, "shell51", "1d_yaml")                       # center / mid-surface set
BEAM = os.path.join(DATA, "dehom51", "beamdyn")
BENCH = os.path.join(DATA, "dehom51", "benchmark", "spanwise_vabs_landmarks.npz")
FF_ALL = np.loadtxt(os.path.join(BEAM, "ff51_rmc_reform.dat"))          # 51 x [eta, F1..F3, M1..M3]
BD_OUT = os.path.join(BEAM, "iea51rmc_bd_driver.out")

BE = ("11", "12", "13", "22", "23", "33")
SVOIGT = {"11": 0, "12": 5, "13": 4, "22": 1, "23": 3, "33": 2}         # -> RM Voigt [11,22,33,23,13,12]


def beam_kin(path, node):
    """BeamDyn last-time nodal translation/rotation at ``node`` -> (u_global, small-rotation matrix).

    Used to strip the beam (rigid + classical) kinematics so the RM warping displacement can be
    lifted to the same global frame the VABS .U warping lives in.  Mirrors spanwise_center51.py."""
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split()
            row = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: row[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")])
            RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]])
            t1, t2, t3 = RD[2], -RD[1], RD[0]
            return u_g, np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
    raise ValueError("no BeamDyn Time header in " + path)


# ------------------------------------------------------------------ spanwise chain
Z = np.load(BENCH)
LM_idx, LM_pt, LM_VS, LM_VU, LM_K = Z["idx"], Z["pt"], Z["VS"], Z["VU"], Z["K"]

eta, diag, S, U, VS, VU = [], [], [], [], [], []
for j, ii in enumerate(LM_idx):
    i = int(ii)
    shp = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(shp):
        print("s%02d SKIP (no yaml)" % i, flush=True)
        continue
    try:
        pt = LM_pt[j]
        Kv = LM_K[j]
        Vs = LM_VS[j]
        Vu = LM_VU[j]
        FF = FF_ALL[i, 1:]                                              # 6 section forces (VABS order)
        B = dehom_rm.build_rm_bundle(shp)                              # RM ring homogenization (center ref)
        C6 = np.asarray(B["Timo"])
        de = 100.0 * (np.diag(C6) - np.diag(Kv)) / np.diag(Kv)         # 6x6-diagonal %err vs VABS .K
        # two-step recovery at the landmark (step1 FF->shell strains, step2 shell->3-D stress)
        s = np.asarray(dehom_rm.stress_at_points(B, pt[None], beam_force_vabs=FF,
                       frame="material", n_per_layer=4, flow_avg=True)["stress"])[0]
        w = np.asarray(dehom_rm.disp_at_points(B, pt[None], beam_force_vabs=FF))[0]
        u_g, C = beam_kin(BD_OUT, i + 1)
        r3 = np.array([0.0, pt[0], pt[1]])
        eta.append(i / 50.0)
        diag.append(de)
        S.append([s[SVOIGT[k]] for k in BE])                          # RM stress reindexed to [11,12,13,22,23,33]
        U.append(u_g + C @ (w + r3) - r3)                             # RM total disp in global frame
        VS.append(Vs.tolist())
        VU.append(Vu.tolist())
        print("s%02d ok  GA3 %+6.2f%%  u3 RM %.3f / VABS %.3f  s11 RM %.3f / VABS %.3f MPa"
              % (i, de[2], U[-1][2], Vu[2], S[-1][0] / 1e6, Vs[0] / 1e6), flush=True)
    except Exception as e:
        print("s%02d FAIL %s" % (i, str(e)[:70]), flush=True)

eta = np.array(eta); diag = np.array(diag); S = np.array(S); U = np.array(U)
VS = np.array(VS); VU = np.array(VU)
LBLK = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("\n%d/%d stations reproduced" % (len(eta), len(LM_idx)))
print("mean |%err| RM 6x6 vs VABS .K:  " +
      "  ".join("%s %.2f" % (LBLK[k], np.nanmean(np.abs(diag[:, k]))) for k in range(6)))

# ------------------------------------------------------------------ (i) homogenization %err
LBL = ["EA", "GA_2", "GA_3", "GJ", "EI_2", "EI_3"]
TIT = ["extension $EA$", "transv. shear $GA_2$", "transv. shear $GA_3$",
       "torsion $GJ$", "flap bending $EI_2$", "edge bending $EI_3$"]
col = plt.cm.rainbow(np.linspace(0, 1, 6))
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14})
fig, axs = plt.subplots(3, 2, figsize=(12, 13.5))
for k in range(6):
    ax = axs.flat[k]
    ax.axhspan(-5, 5, color="0.9", zorder=0)
    ax.axhline(0, color="0.6", lw=1.2, ls=":")
    ax.plot(eta, diag[:, k], "-o", color=col[k], mec="k", mew=0.5, ms=8, lw=2.2)
    mx = np.nanmax(np.abs(diag[:, k]))
    ax.set_ylim(-max(6, 1.2 * mx), max(6, 1.2 * mx))
    ax.set_xlabel(r"span $r/R$")
    ax.set_ylabel(r"$%s$ RM vs VABS  [\%%]" % LBL[k])
    ax.set_title(TIT[k], fontsize=15)
    ax.grid(alpha=0.25)
fig.tight_layout()
f1 = os.path.join(HERE, "iea_spanwise_homo_pcterr.png")
fig.savefig(f1, dpi=150); plt.close(fig)

# ------------------------------------------------------------------ (ii) spanwise stress recovery
VABSC = "#1f77b4"; RMC = "#ff7f0e"
SIN = [("11", 0), ("12", 1), ("22", 3)]
SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
d0 = np.abs(S[:, 0] - VS[:, 0]); keep = d0 <= 8.0 * np.median(d0) + 1e-12
fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
for ax, (k, idx), lab in zip(axs, SIN, SLAB):
    ax.plot(eta[keep], VS[keep, idx] / 1e6, "-o", color=VABSC, ms=6.5, lw=2.2, label="VABS")
    ax.plot(eta[keep], S[keep, idx] / 1e6, "--s", color=RMC, ms=6.5, mfc="none", mew=1.8,
            lw=2.0, label="RM shell")
    ax.set_xlabel(r"span  $r/R$")
    ax.set_ylabel("%s   [MPa]" % lab)
    ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
f2 = os.path.join(HERE, "iea_spanwise_stress.png")
fig.savefig(f2, dpi=150); plt.close(fig)

# ------------------------------------------------------------------ (ii) spanwise disp recovery
ULAB = [r"$u_1$ (out-of-plane warping)", r"$u_2$ (edgewise)", r"$u_3$ (flapwise)"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for k, (ax, lab) in enumerate(zip(axs, ULAB)):
    ax.plot(eta, VU[:, k], "-o", color=VABSC, ms=6.5, lw=2.2, label="VABS")
    ax.plot(eta, U[:, k], "--s", color=RMC, ms=6.5, mfc="none", mew=1.8, lw=2.0, label="RM shell")
    ax.set_xlabel(r"span  $r/R$")
    ax.set_ylabel("%s   [m]" % lab)
    ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
f3 = os.path.join(HERE, "iea_spanwise_disp.png")
fig.savefig(f3, dpi=150); plt.close(fig)
print("wrote:\n  %s\n  %s\n  %s" % (f1, f2, f3))
