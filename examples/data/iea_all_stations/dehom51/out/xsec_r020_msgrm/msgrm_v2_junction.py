"""msgrm_v2_junction.py -- SECOND-ORDER (V2, Eq. 64-66) MSG-RM dehom of station r=0.2,
with the cap/web JUNCTION thickness path as the focus.

Pipeline is the systematic 1D-yaml route end to end:
    iea_s10_shell.yaml -> dehom_rm.build_rm_bundle (ring homo, wall G = core rm_plate_msg,
    g_source="msg") -> core msgrm_strain_at_depth for the local recovery, FIRST order
    (dE1 span + dE2 arc, as the production msgrm_dehom_r020.py) and SECOND order
    (adds dE11 span-span from st_cl2, dE12 span-arc, dE22 arc-arc -- both arc derivatives
    layup-boundary-aware, same nodal scheme as dE2).

Outputs: full-section rms metrics vs VABS .SM for BOTH orders (now including the
transverse sigma33/sigma13/sigma23 that only V2 can carry), the junction path
(iea_s10.lp_sparcap_left_thickness.coords) table + figure, and data npz.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain
from msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from segment_indep import quad_ops_indep
from opensg_jax.fe_jax.msg_materials import rotation_6x6
from opensg_jax.fe_jax.msg_dehom import _macro_recovery

SHELL = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]
JPATH = os.path.join(IEA, "coords", "iea_s10.lp_sparcap_left_thickness.coords")
DATA = os.path.join(HERE, "data"); FIG = os.path.join(HERE, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ================= bundle: 1D-yaml -> ring homo with the core MSG G =================
t0 = time.perf_counter()
B = dehom_rm.build_rm_bundle(SHELL)                       # g_source="msg" (core rm_plate_msg)
print("bundle built (g_source=%s) %.1fs" % (B.get("g_source"), time.perf_counter() - t0), flush=True)
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
C6 = np.asarray(B["Timo"])
_, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
nodes_s, quads_s, _hs = B["strip"]
n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
frac = float(B.get("frac", 0.0))

A0 = corners[rc[:, 0]]; T = corners[rc[:, 1]] - corners[rc[:, 0]]
L2 = (T ** 2).sum(1); tun = T / np.sqrt(L2)[:, None]
nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
flip = ((cen - emid) * nvec).sum(1) < 0
nvec[flip] *= -1.0
Lel = np.sqrt(L2)


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


# ================= per-element strains + boundary-aware gradient LADDER =================
s6mid = np.zeros((n_el, 6)); dE1e = np.zeros((n_el, 6)); dE11e = np.zeros((n_el, 6))
for e in range(n_el):
    s6, _ = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    s6mid[e] = np.asarray(s6, float)
    Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
    BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]), B["cross"], B["ax"])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)
    # span-span gradient: the macro-strain second recovery (the warping-gradient
    # counterpart of the BDh term would need the NEXT warping order; for the
    # constant-force blade case st_cl2 dominates and the remainder is higher order)
    dE11e[e] = np.asarray(BDe @ st_cl2, float)

deg = np.zeros(n_nd, int)
nd_el = [[] for _ in range(n_nd)]
for e in range(n_el):
    for nd in (int(rc[e, 0]), int(rc[e, 1])):
        deg[nd] += 1; nd_el[nd].append(e)


def arc_derivative(field_e):
    """layup-boundary-aware arc derivative of an elementwise 6-field (nodal-average
    only where exactly two SAME-layup elements meet -- the production dE2 scheme)."""
    fn = np.full((n_nd, 6), np.nan)
    for nd in range(n_nd):
        if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
            fn[nd] = 0.5 * (field_e[nd_el[nd][0]] + field_e[nd_el[nd][1]])
    out = np.zeros((n_el, 6))
    for e in range(n_el):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        v0 = fn[c0] if np.isfinite(fn[c0, 0]) else field_e[e]
        v1 = fn[c1] if np.isfinite(fn[c1, 0]) else field_e[e]
        out[e] = (v1 - v0) / Lel[e]
    return out, fn


dE2e, s6n = arc_derivative(s6mid)
dE12e, _ = arc_derivative(dE1e)               # span-arc
dE22e, _ = arc_derivative(dE2e)               # arc-arc (second)
print("gradient magnitudes (rms): |E| %.3e |E,1| %.3e |E,2| %.3e |E,11| %.3e |E,12| %.3e |E,22| %.3e"
      % tuple(float(np.sqrt(np.mean(x ** 2))) for x in
              (s6mid, dE1e, dE2e, dE11e, dE12e, dE22e)), flush=True)

# ================= VABS reference + projections + web registration =================
dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
elg, xig, zdg = fast_project(sm_xy)

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

# ================= core homo per laminate (V2 columns come along for free) =================
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb, fraction=frac)
         for ln, i in ldb.items()}
print("core rm_plate_msg homo: %d laminates; Ustar_rel range %.1e..%.1e" %
      (len(warpM), min(w["Ustar_rel"] for w in warpM.values()),
       max(w["Ustar_rel"] for w in warpM.values())), flush=True)


def recover(ip_el, ip_xi, ip_z, second):
    e = ip_el; ln = layups[e]
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    s6 = s6mid[e].copy()
    for row in (2, 5):
        v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
        v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
        s6[row] = (1.0 - ip_xi) * v0 + ip_xi * v1
    kw = dict(dE11=dE11e[e], dE12=dE12e[e], dE22=dE22e[e]) if second else {}
    Gam, Sig, ply = msgrm_strain_at_depth(warpM[ln], ip_z, s6, dE1e[e], dE2e[e], **kw)
    return rotation_6x6(-ply) @ np.asarray(Sig, float)


# ================= full-section: first vs second order vs VABS =================
t0 = time.perf_counter()
P = len(sm_xy)
sM1 = np.zeros((P, 6)); sM2 = np.zeros((P, 6))
for ip in range(P):
    z = zdg[ip] - zoff[elg[ip]]
    sM1[ip] = recover(elg[ip], xig[ip], z, False)
    sM2[ip] = recover(elg[ip], xig[ip], z, True)
sM1 /= 1e6; sM2 /= 1e6
print("recovery at %d gauss pts (both orders): %.1fs" % (P, time.perf_counter() - t0), flush=True)

webp = is_web[elg]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


rep = ["=== r=0.2 MSG-RM dehom, FIRST vs SECOND order (V2) vs VABS ===",
       "rms diff vs VABS (MPa), WEB / SKIN:"]
names = ["s11", "s22", "s33", "s23", "s13", "s12"]
for ci in (0, 1, 5, 2, 4, 3):
    rep.append("  %-4s  first %7.3f / %7.3f   V2 %7.3f / %7.3f   (VABS rms %7.3f / %7.3f)"
               % (names[ci],
                  rms(sVg[webp, ci] - sM1[webp, ci]), rms(sVg[~webp, ci] - sM1[~webp, ci]),
                  rms(sVg[webp, ci] - sM2[webp, ci]), rms(sVg[~webp, ci] - sM2[~webp, ci]),
                  rms(sVg[webp, ci]), rms(sVg[~webp, ci])))
print("\n".join(rep), flush=True)

# ================= the JUNCTION thickness path =================
jc = np.loadtxt(JPATH)
jxy = jc[:, :2]; jarc = jc[:, 2]
elj, xij, zdj = fast_project(jxy)
d2near = ((sm_xy[None, :, :] - jxy[:, None, :]) ** 2).sum(2)
near = np.argmin(d2near, 1)
dist = np.sqrt(d2near[np.arange(len(jxy)), near])
sJ1 = np.zeros((len(jxy), 6)); sJ2 = np.zeros((len(jxy), 6))
for k in range(len(jxy)):
    z = zdj[k] - zoff[elj[k]]
    sJ1[k] = recover(elj[k], xij[k], z, False) / 1e6
    sJ2[k] = recover(elj[k], xij[k], z, True) / 1e6
sJV = sVg[near]

jrep = ["", "=== JUNCTION path (lp_sparcap_left_thickness, %d pts; nearest .SM gauss "
        "dist max %.2f mm) ===" % (len(jxy), 1e3 * dist.max()),
        "rms diff vs VABS along the path (MPa):"]
for ci in (0, 1, 5, 2, 4, 3):
    jrep.append("  %-4s  first %8.3f   V2 %8.3f   (VABS rms %8.3f)"
                % (names[ci], rms(sJV[:, ci] - sJ1[:, ci]), rms(sJV[:, ci] - sJ2[:, ci]),
                   rms(sJV[:, ci])))
print("\n".join(jrep), flush=True)

# path figure: in-plane s11 + the transverse pair s13, s33 (V2 payload)
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
for a_, ci in zip(ax, (0, 4, 2)):
    a_.plot(jarc, sJV[:, ci], "k-", lw=2, label="VABS")
    a_.plot(jarc, sJ1[:, ci], "s--", color="#1f77b4", ms=5, label="MSG-RM 1st order")
    a_.plot(jarc, sJ2[:, ci], "o-", color="#ff7f0e", ms=5, mfc="none", label="MSG-RM V2 (Eq. 66)")
    a_.set_xlabel("thickness position [mm]")
    a_.set_ylabel(r"$\sigma_{%s}$ [MPa]" % {0: "11", 4: "13", 2: "33"}[ci])
    a_.grid(alpha=0.3)
ax[0].legend(frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r020_junction_v2.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez(os.path.join(DATA, "msgrm_v2_junction.npz"),
         sm_xy=sm_xy, s_first=sM1, s_v2=sM2, s_vabs=sVg, is_web_gauss=webp,
         j_xy=jxy, j_arc=jarc, sJ_first=sJ1, sJ_v2=sJ2, sJ_vabs=sJV,
         dE11e=dE11e, dE12e=dE12e, dE22e=dE22e, C6=C6, FF=FF)
open(os.path.join(DATA, "README_v2_junction.txt"), "w").write("\n".join(rep + jrep) + "\n")
print("saved data/msgrm_v2_junction.npz + figures/r020_junction_v2.png", flush=True)
