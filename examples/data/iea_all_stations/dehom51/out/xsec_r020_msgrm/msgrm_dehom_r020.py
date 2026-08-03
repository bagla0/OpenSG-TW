"""msgrm_dehom_r020.py -- FULL MSG-RM dehomogenization of station r=0.2 (iea_s10, mid-ref).

The production MSG-RM pipeline run, stored in THIS folder (xsec_r020_msgrm):
  * ring homogenization with the MSG (Yu-2002 LS) wall G      (dehom_rm g_source="msg" default)
  * web through-thickness REGISTRATION offset (physical wall centre from the .sg gauss cloud)
  * MSG-RM first-order recovery (gradient warping, dE1 span + dE2 arc) with LAYUP-BOUNDARY-AWARE
    arc gradients (no differencing across section changes -> no cap/panel transition spikes)
  * local STRESS at every VABS .SM gauss point  + local DISP (warping) at every VABS .U node
  * comparison vs VABS and vs the PREVIOUS Whitney-G pipeline (xsec_r020_dehom/_rm_s10_cache.npz)
Outputs here: data/ (npz + README), figures/ (line + contour plots), msgrm_dehom.log.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
OLD = os.path.join(IEA, "out", "xsec_r020_dehom")
sys.path.insert(0, XSEC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax

jax.config.update("jax_enable_x64", True)
import yaml as _yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain
from msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from segment_indep import quad_ops_indep
from opensg_jax.fe_jax.msg_materials import rotation_6x6

SHELL = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]
BD_VABS = os.path.join(VABS, "iea51vabs_bd_driver.out")
DATA = os.path.join(HERE, "data"); FIG = os.path.join(HERE, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)
BLUE = "#1f77b4"; ORANGE = "#ff7f0e"; GRAY = "0.55"
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 14,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})

# ================= bundle (MSG G is the default now) =================
t0 = time.perf_counter()
B = dehom_rm.build_rm_bundle(SHELL)                      # g_source="msg"
print("bundle built (g_source=%s) %.1fs" % (B.get("g_source"), time.perf_counter() - t0), flush=True)
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
C6 = np.asarray(B["Timo"])
from opensg_jax.fe_jax.msg_dehom import _macro_recovery

_, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
nodes_s, quads_s, _hs = B["strip"]
n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
frac = float(B.get("frac", 0.0))
hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}

# ================= fast vectorized point projection =================
A0 = corners[rc[:, 0]]; T = corners[rc[:, 1]] - corners[rc[:, 0]]
L2 = (T ** 2).sum(1); tun = T / np.sqrt(L2)[:, None]
nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
flip = ((cen - emid) * nvec).sum(1) < 0
nvec[flip] *= -1.0


def fast_project(P, chunk=4000):
    P = np.atleast_2d(np.asarray(P, float))
    el = np.zeros(len(P), int); xi = np.zeros(len(P)); zd = np.zeros(len(P))
    for i0 in range(0, len(P), chunk):
        Pc = P[i0:i0 + chunk]
        dp = Pc[:, None, :] - A0[None, :, :]
        tpar = np.clip((dp * T[None]).sum(2) / L2[None], 0.0, 1.0)
        foot = A0[None] + tpar[..., None] * T[None]
        d2 = ((Pc[:, None, :] - foot) ** 2).sum(2)
        e = np.argmin(d2, 1)
        r_ = np.arange(len(Pc))
        el[i0:i0 + chunk] = e
        xi[i0:i0 + chunk] = tpar[r_, e]
        zd[i0:i0 + chunk] = ((Pc - foot[r_, e]) * nvec[e]).sum(1)
    return el, xi, zd


# ================= per-element strains + boundary-aware gradients =================
s6mid = np.zeros((n_el, 6)); dE1e = np.zeros((n_el, 6))
for e in range(n_el):
    s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    s6mid[e] = np.asarray(s6, float)
    Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
    BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]), B["cross"], B["ax"])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)

deg = np.zeros(n_nd, int)
nd_el = [[] for _ in range(n_nd)]
for e in range(n_el):
    for nd in (int(rc[e, 0]), int(rc[e, 1])):
        deg[nd] += 1; nd_el[nd].append(e)
# nodal average of s6 ONLY where exactly two SAME-LAYUP elements meet
s6n = np.full((n_nd, 6), np.nan)
for nd in range(n_nd):
    if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
        s6n[nd] = 0.5 * (s6mid[nd_el[nd][0]] + s6mid[nd_el[nd][1]])
Lel = np.sqrt(L2)
dE2e = np.zeros((n_el, 6))
for e in range(n_el):
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    v0 = s6n[c0] if np.isfinite(s6n[c0, 0]) else s6mid[e]
    v1 = s6n[c1] if np.isfinite(s6n[c1, 0]) else s6mid[e]
    dE2e[e] = (v1 - v0) / Lel[e]

# ================= VABS reference fields =================
dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
U = np.loadtxt(os.path.join(VABS, "iea_s10.sg.U")); U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]; uV_tot = U[:, 3:6]


def beam_kinematics(path, node):
    Lf = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(Lf):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in Lf[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
            return u_g, C
    raise ValueError("no BeamDyn header")


u_g, Cbk = beam_kinematics(BD_VABS, 11)
r3 = np.column_stack([np.zeros(len(xy)), xy[:, 0], xy[:, 1]])
uV = ((np.linalg.inv(Cbk) @ (uV_tot - u_g + r3).T).T - r3) * 1e3          # VABS warping (mm)

# ================= projections + web registration =================
t0 = time.perf_counter()
elg, xig, zdg = fast_project(sm_xy)
eln, xin, zdn = fast_project(xy)
print("vectorized projection: %d gauss + %d nodes in %.1fs" % (len(sm_xy), len(xy), time.perf_counter() - t0), flush=True)

adjm = {}
for e, (a, b) in enumerate(rc):
    adjm.setdefault(a, []).append((b, e)); adjm.setdefault(b, []).append((a, e))
junc = set(np.where(deg >= 3)[0])
is_web = np.zeros(n_el, bool)
seen = set()
for j in junc:
    for (nxt, e0) in adjm[j]:
        if e0 in seen:
            continue
        chain, prev, cur = [e0], j, nxt
        seen.add(e0)
        while cur not in junc and deg[cur] == 2:
            (n1, e1), (n2_, e2) = adjm[cur][0], adjm[cur][1]
            nn2, ee = (n1, e1) if n1 != prev else (n2_, e2)
            if ee in seen:
                break
            chain.append(ee); seen.add(ee)
            prev, cur = cur, nn2
        if cur in junc:
            arc = sum(Lel[c] for c in chain)
            cv = corners[cur] - corners[j]; ch = float(np.linalg.norm(cv))
            if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                is_web[chain] = True
zoff = np.zeros(n_el)
for e in np.where(is_web)[0]:
    zz = zdg[elg == e]
    if len(zz) >= 8:
        zoff[e] = 0.5 * (np.percentile(zz, 2) + np.percentile(zz, 98))
print("webs: %d elems, |zoff| mean %.1f mm" % (is_web.sum(), 1e3 * np.abs(zoff[is_web]).mean()), flush=True)

# ================= MSG-RM stress at gauss points =================
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb,
         fraction=frac) for ln, i in ldb.items()}
t0 = time.perf_counter()
P = len(sm_xy)
sM = np.zeros((P, 6))
for ip in range(P):
    e = elg[ip]; xi = xig[ip]; ln = layups[e]
    z = zdg[ip] - zoff[e]
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    s6 = s6mid[e].copy()
    for row in (2, 5):
        v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
        v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
        s6[row] = (1.0 - xi) * v0 + xi * v1
    Gam, Sig, ply = msgrm_strain_at_depth(warpM[ln], z, s6, dE1e[e], dE2e[e])
    sM[ip] = rotation_6x6(-ply) @ np.asarray(Sig, float)
sM /= 1e6
t_s = time.perf_counter() - t0
print("MSG-RM stress at %d gauss pts: %.1fs" % (P, t_s), flush=True)

# ================= MSG-RM disp (warping) at .U nodes =================
t0 = time.perf_counter()
wn = np.asarray(aA).reshape(-1, 6)
uM = np.zeros((len(xy), 3))
for ip in range(len(xy)):
    e = eln[ip]; xi = xin[ip]
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    umid = (1.0 - xi) * wn[c0, 0:3] + xi * wn[c1, 0:3]
    om = (1.0 - xi) * wn[c0, 3:6] + xi * wn[c1, 3:6]
    z = zdn[ip] - zoff[e]
    n2, n3 = nvec[e]
    uM[ip] = umid + z * np.cross(om, np.array([0.0, n2, n3]))
uM *= 1e3
t_u = time.perf_counter() - t0
print("MSG-RM warping disp at %d nodes: %.1fs" % (len(xy), t_u), flush=True)

# ================= previous Whitney pipeline + metrics =================
old = np.load(os.path.join(OLD, "_rm_s10_cache.npz"))
sW = old["sRg"]; uW = old["uR"]
webp = is_web[elg]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


rep = []
rep.append("=== r=0.2 MSG-RM dehom vs VABS (and previous Whitney pipeline) ===")
rep.append("STRESS rms diff vs VABS (MPa):        WEB      SKIN")
for tag, s in (("Whitney (previous pipeline)", sW), ("MSG-RM (this run)", sM)):
    rep.append("  %-28s  s11 %7.2f / %7.2f   s12 %7.2f / %7.2f"
               % (tag, rms(sVg[webp, 0] - s[webp, 0]), rms(sVg[~webp, 0] - s[~webp, 0]),
                  rms(sVg[webp, 5] - s[webp, 5]), rms(sVg[~webp, 5] - s[~webp, 5])))
rep.append("  VABS signal rms:              s11 %7.2f / %7.2f   s12 %7.2f / %7.2f"
           % (rms(sVg[webp, 0]), rms(sVg[~webp, 0]), rms(sVg[webp, 5]), rms(sVg[~webp, 5])))
rep.append("DISP (warping) rms diff vs VABS (mm), u1/u2/u3:")
for tag, u in (("Whitney (previous)", uW), ("MSG-RM (this run)", uM)):
    rep.append("  %-28s  %7.4f  %7.4f  %7.4f   (VABS rms %7.4f %7.4f %7.4f)"
               % (tag, rms(uV[:, 0] - u[:, 0]), rms(uV[:, 1] - u[:, 1]), rms(uV[:, 2] - u[:, 2]),
                  rms(uV[:, 0]), rms(uV[:, 1]), rms(uV[:, 2])))
rep.append("Timo diag (MSG G): " + np.array2string(np.diag(C6), precision=4))
rep.append("timing: stress %.1fs, disp %.1fs" % (t_s, t_u))
print("\n".join(rep), flush=True)
open(os.path.join(DATA, "README_metrics.txt"), "w").write("\n".join(rep) + "\n")

np.savez(os.path.join(DATA, "msgrm_r020_fields.npz"),
         sm_xy=sm_xy, stress_msgrm=sM, stress_whitney=sW, stress_vabs=sVg,
         xy_nodes=xy, disp_msgrm=uM, disp_whitney=uW, disp_vabs=uV,
         zoff=zoff, is_web=is_web, el_gauss=elg, xi_gauss=xig, z_gauss=zdg,
         el_node=eln, z_node=zdn, C6_msg=C6, FF=FF)
print("data saved to data/msgrm_r020_fields.npz", flush=True)

# ================= mesh for contours =================
Lf = [l for l in open(os.path.join(VABS, "iea_s10.sg")).read().splitlines() if l.strip()]
hi = next(i for i, l in enumerate(Lf) if len(l.split()) == 3
          and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
nn, ne, nm = [int(x) for x in Lf[hi].split()]
conn = [[int(x) for x in Lf[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    if len(c0) == 3:
        trl.append(c0[:3])
    elif len(c0) >= 4:
        trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]])
tris = np.array(trl); M = tris.shape[0]
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
gxy = sm_xy.reshape(M, 3, 2)
Ai = np.linalg.inv(np.concatenate([np.ones((M, 3, 1)), gxy], 2))
Cc = np.concatenate([np.ones((M, 3, 1)), xy[tris]], 2)
EP = xy[tris].reshape(-1, 2)
etri = mtri.Triangulation(EP[:, 0], EP[:, 1], np.arange(3 * M).reshape(M, 3))


def corners_of(val):
    g = val.reshape(M, 3, 1)
    c = Cc @ (Ai @ g)
    return np.clip(c, g.min(1, keepdims=True), g.max(1, keepdims=True)).reshape(-1)


# ---- stress contours (exploded): VABS | MSG-RM ----
fig, ax = plt.subplots(3, 2, figsize=(13.5, 12.0))
for r_, (ci, lab) in enumerate([(0, r"$\sigma_{11}$"), (1, r"$\sigma_{22}$"), (5, r"$\sigma_{12}$")]):
    mlim = np.nanpercentile(np.abs(np.r_[sVg[:, ci], sM[:, ci]]), 99) or 1e-9
    for c_, val in enumerate([sVg[:, ci], sM[:, ci]]):
        cs = ax[r_, c_].tripcolor(etri, np.clip(corners_of(val), -mlim, mlim), shading="gouraud",
                                  cmap="rainbow", vmin=-mlim, vmax=mlim)
        ax[r_, c_].set_aspect("equal"); ax[r_, c_].axis("off")
    ax[r_, 0].text(-0.05, 0.5, lab, transform=ax[r_, 0].transAxes, rotation=90,
                   va="center", ha="right", fontsize=20)
    cb = fig.colorbar(cs, ax=ax[r_, :].tolist(), shrink=0.85, pad=0.015)
    cb.set_label("MPa", fontsize=18); cb.ax.tick_params(labelsize=15)
ax[0, 0].set_title("VABS", fontsize=20, pad=8); ax[0, 1].set_title("MSG-RM shell", fontsize=20, pad=8)
fig.savefig(os.path.join(FIG, "r020_msgrm_stress_contours.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- disp contours (nodal): VABS | MSG-RM ----
fig, ax = plt.subplots(3, 2, figsize=(13.5, 12.0))
for r_, (ci, lab) in enumerate([(0, r"$u_1$ (out-of-plane warping)"), (1, r"$u_2$ (edgewise)"),
                                (2, r"$u_3$ (flapwise)")]):
    mlim = np.nanpercentile(np.abs(np.r_[uV[:, ci], uM[:, ci]]), 99) or 1e-9
    for c_, val in enumerate([uV[:, ci], uM[:, ci]]):
        cs = ax[r_, c_].tripcolor(tri, np.clip(val, -mlim, mlim), shading="gouraud",
                                  cmap="rainbow", vmin=-mlim, vmax=mlim)
        ax[r_, c_].set_aspect("equal"); ax[r_, c_].axis("off")
    ax[r_, 0].text(-0.05, 0.5, lab, transform=ax[r_, 0].transAxes, rotation=90,
                   va="center", ha="right", fontsize=15)
    cb = fig.colorbar(cs, ax=ax[r_, :].tolist(), shrink=0.85, pad=0.015)
    cb.set_label("mm", fontsize=18); cb.ax.tick_params(labelsize=15)
ax[0, 0].set_title("VABS", fontsize=20, pad=8); ax[0, 1].set_title("MSG-RM shell", fontsize=20, pad=8)
fig.savefig(os.path.join(FIG, "r020_msgrm_disp_contours.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- line plots ----
skin = np.where(~is_web)[0]
ang = np.arctan2(emid[skin, 1] - cen[1], emid[skin, 0] - cen[0])
order = skin[np.argsort(ang)]
s_arc = np.cumsum(Lel[order]) - 0.5 * Lel[order]; s_arc /= s_arc.max()


def elem_mean(field, els):
    out = np.full(len(order), np.nan)
    for i, e in enumerate(order):
        m = els == e
        if m.any():
            out[i] = field[m].mean()
    return out


fig, ax = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True)
for r_, (ci, lab) in enumerate([(0, r"$\sigma_{11}$ (MPa)"), (5, r"$\sigma_{12}$ (MPa)")]):
    ax[r_].plot(s_arc, elem_mean(sVg[:, ci], elg), "-", color=BLUE, lw=2.2, label="VABS")
    ax[r_].plot(s_arc, elem_mean(sW[:, ci], elg), "-", color=GRAY, lw=1.1, label="Whitney (previous)")
    ax[r_].plot(s_arc, elem_mean(sM[:, ci], elg), "--s", color=ORANGE, ms=4.5, mfc="none", lw=1.4,
                markevery=4, label="MSG-RM")
    ax[r_].set_ylabel(lab); ax[r_].grid(alpha=0.3)
ax[1].set_xlabel(r"normalized skin arc $s/S$")
ax[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "r020_msgrm_circ_stress.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(3, 1, figsize=(13.5, 11.5), sharex=True)
for r_, (ci, lab) in enumerate([(0, r"$u_1$ (mm)"), (1, r"$u_2$ (mm)"), (2, r"$u_3$ (mm)")]):
    ax[r_].plot(s_arc, elem_mean(uV[:, ci], eln), "-", color=BLUE, lw=2.2, label="VABS")
    ax[r_].plot(s_arc, elem_mean(uW[:, ci], eln), "-", color=GRAY, lw=1.1, label="Whitney (previous)")
    ax[r_].plot(s_arc, elem_mean(uM[:, ci], eln), "--s", color=ORANGE, ms=4.5, mfc="none", lw=1.4,
                markevery=4, label="MSG-RM")
    ax[r_].set_ylabel(lab); ax[r_].grid(alpha=0.3)
ax[2].set_xlabel(r"normalized skin arc $s/S$")
ax[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "r020_msgrm_circ_disp.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# cap through-thickness
capels = [e for e in skin if abs(emid[e, 0]) < 0.35 and emid[e, 1] > 0]
ecap = capels[len(capels) // 2]
m = elg == ecap
h = hth[layups[ecap]]
zeta = np.clip((zdg[m] + h / 2) / h, -0.1, 1.1)
o = np.argsort(zeta)
fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2))
for k, (ci, lab) in enumerate([(0, r"$\sigma_{11}$ (MPa)"), (5, r"$\sigma_{12}$ (MPa)")]):
    ax[k].plot(zeta[o], sVg[m, ci][o], "o-", color=BLUE, ms=6.5, lw=1.4, label="VABS")
    ax[k].plot(zeta[o], sM[m, ci][o], "--s", color=ORANGE, ms=7.0, mfc="none", lw=1.4, label="MSG-RM")
    ax[k].set_xlabel(r"$\zeta$ (0=OML, 1=IML), cap centre"); ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "r020_msgrm_cap_tt.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# web through-thickness
chains_w, seen2 = [], set()
for j in junc:
    for (nxt, e0) in adjm[j]:
        if e0 in seen2:
            continue
        chain, prev, cur = [e0], j, nxt
        seen2.add(e0)
        while cur not in junc and deg[cur] == 2:
            (n1, e1), (n2_, e2) = adjm[cur][0], adjm[cur][1]
            nn2, ee = (n1, e1) if n1 != prev else (n2_, e2)
            if ee in seen2:
                break
            chain.append(ee); seen2.add(ee)
            prev, cur = cur, nn2
        if cur in junc and is_web[chain].all() and len(chain) > 3:
            chains_w.append(chain)
chains_w = sorted(chains_w, key=lambda c: emid[c].mean(0)[0])
fig, ax = plt.subplots(1, len(chains_w), figsize=(6.2 * len(chains_w), 5.2))
ax = np.atleast_1d(ax)
for wi, chain in enumerate(chains_w):
    ce = np.array(chain)
    m = np.isin(elg, ce)
    y3 = sm_xy[m, 1]; ymid = 0.5 * (y3.min() + y3.max()); span = y3.max() - y3.min()
    mm = m.copy(); mm[m] = np.abs(y3 - ymid) < 0.12 * span
    e_of = elg[mm]
    hh = np.array([hth[layups[e]] for e in e_of])
    zeta = np.clip((zdg[mm] - zoff[e_of] + hh / 2) / hh, -0.15, 1.15)
    o = np.argsort(zeta)
    ax[wi].plot(zeta[o], sVg[mm, 5][o], "o", color=BLUE, ms=6.5, label="VABS")
    ax[wi].plot(zeta[o], sW[mm, 5][o], "x", color=GRAY, ms=6.0, label="Whitney (previous)")
    ax[wi].plot(zeta[o], sM[mm, 5][o], "s", mfc="none", color=ORANGE, ms=7.0, label="MSG-RM")
    ax[wi].set_xlabel(r"$\zeta$ through web %d" % wi)
    ax[wi].set_ylabel(r"$\sigma_{12}$ (MPa)" if wi == 0 else ""); ax[wi].grid(alpha=0.3)
ax[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "r020_msgrm_web_tt.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("all figures written to figures/", flush=True)
