"""msgrm_dehom_r020_oml.py -- r=0.2 MSG-RM dehomogenization at the OML laminate reference.

Same production pipeline as xsec_r020_msgrm/msgrm_dehom_r020.py but with the SINGLE
initial-stage reference argument set to "oml": the OML-built 1-D yaml
(shell51/1d_yaml_oml/iea_s10_shell.yaml, contour ON the outer mold line, plies stacked
inward), which via build_rm_bundle drives ring reference, ABD, MSG G / recovery-warping
z_ref (=0, the OML face) and the depth convention consistently.

Registration: per-element face offset zoff_face[e] = p2-percentile of the element's VABS
gauss depth cloud -- aligns the plate-SG z=0 with the PHYSICAL outer face of that wall
(skin: ~0 by construction since the contour IS the OML; webs: absorbs the mid-line
stacking offset of the .sg, the same physical correction as the center run's mid-span
registration).

Outputs (this folder): data/msgrm_r020_oml_fields.npz + README_oml_metrics.txt,
figures/dehom_r020_{circ,cap}_{stress,disp}_oml.png (CPB line style, total disp in m),
figures/r020_row_{S11,S22,S12,u1,u2,u3}_oml.png (contour rows).
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))                 # dehom51
ROOT = os.path.abspath(os.path.join(IEA, ".."))                       # iea_all_stations
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
MSGRM = os.path.join(IEA, "out", "xsec_r020_msgrm")
sys.path.insert(0, XSEC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import ScalarFormatter

import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain
from msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from segment_indep import quad_ops_indep
from opensg_jax.fe_jax.msg_dehom import _macro_recovery
from opensg_jax.fe_jax.msg_materials import rotation_6x6

SHELL = os.path.join(ROOT, "shell51", "1d_yaml_oml", "iea_s10_shell.yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
VBD = os.path.join(IEA, "out", "dehom_vabs")
FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]
BD_VABS = os.path.join(VABS, "iea51vabs_bd_driver.out")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
VABSC = "#1f77b4"
RMC = "#ff7f0e"

# ================= bundle at the OML reference =================
t0 = time.perf_counter()
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")
frac = float(B.get("frac", 0.0))
assert frac == 0.0, "OML bundle must carry frac=0"
print("OML bundle built (ref=%s frac=%.1f g=%s) %.1fs"
      % (B.get("ref"), frac, B.get("g_source"), time.perf_counter() - t0), flush=True)
print("Timo diag (OML, MSG G):", np.array2string(np.diag(np.asarray(B["Timo"])), precision=4), flush=True)
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
C6 = np.asarray(B["Timo"])
_, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
nodes_s, quads_s, _hs = B["strip"]
n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}

# ================= fast vectorized point projection =================
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
s6n = np.full((n_nd, 6), np.nan)
for nd in range(n_nd):
    if deg[nd] == 2 and layups[nd_el[nd][0]] == layups[nd_el[nd][1]]:
        s6n[nd] = 0.5 * (s6mid[nd_el[nd][0]] + s6mid[nd_el[nd][1]])
dE2e = np.zeros((n_el, 6))
for e in range(n_el):
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    v0 = s6n[c0] if np.isfinite(s6n[c0, 0]) else s6mid[e]
    v1 = s6n[c1] if np.isfinite(s6n[c1, 0]) else s6mid[e]
    dE2e[e] = (v1 - v0) / Lel[e]

# ================= VABS reference fields (reuse the production npz) =================
zc = np.load(os.path.join(MSGRM, "data", "msgrm_r020_fields.npz"))
sm_xy = zc["sm_xy"]; sVg = zc["stress_vabs"]                        # MPa
xy = zc["xy_nodes"]; uV = zc["disp_vabs"]                           # warping, mm
sC = zc["stress_msgrm"]; uC = zc["disp_msgrm"]                      # center-ref MSG-RM run

# ================= projections + face registration =================
t0 = time.perf_counter()
elg, xig, zdg = fast_project(sm_xy)
eln, xin, zdn = fast_project(xy)
print("projection: %d gauss + %d nodes in %.1fs" % (len(sm_xy), len(xy), time.perf_counter() - t0), flush=True)

# web detection (same chain walk as the center run) for the metrics split
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

# face registration: SG z=0 must sit on the PHYSICAL outer face of each wall
zoff_face = np.zeros(n_el)
for e in range(n_el):
    zz = zdg[elg == e]
    if len(zz) >= 8:
        zoff_face[e] = np.percentile(zz, 2)
print("face registration: skin |zoff| mean %.2f mm, web |zoff| mean %.1f mm"
      % (1e3 * np.abs(zoff_face[~is_web]).mean(), 1e3 * np.abs(zoff_face[is_web]).mean()), flush=True)

# ================= MSG-RM stress at gauss points (OML SG, z_ref=0) =================
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb,
         fraction=frac) for ln, i in ldb.items()}
t0 = time.perf_counter()
P = len(sm_xy)
sM = np.zeros((P, 6))
for ip in range(P):
    e = elg[ip]; xi = xig[ip]; ln = layups[e]
    z = zdg[ip] - zoff_face[e]
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
print("MSG-RM stress at %d gauss pts (OML): %.1fs" % (P, t_s), flush=True)

# ================= MSG-RM warping disp at .U nodes =================
t0 = time.perf_counter()
wn = np.asarray(aA).reshape(-1, 6)
uM = np.zeros((len(xy), 3))
for ip in range(len(xy)):
    e = eln[ip]; xi = xin[ip]
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    umid = (1.0 - xi) * wn[c0, 0:3] + xi * wn[c1, 0:3]
    om = (1.0 - xi) * wn[c0, 3:6] + xi * wn[c1, 3:6]
    z = zdn[ip]
    n2, n3 = nvec[e]
    uM[ip] = umid + z * np.cross(om, np.array([0.0, n2, n3]))
uM *= 1e3
t_u = time.perf_counter() - t0
print("MSG-RM warping disp at %d nodes (OML): %.1fs" % (len(xy), t_u), flush=True)

# ================= metrics =================
webp = is_web[elg]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


rep = ["=== r=0.2 MSG-RM dehom, OML reference (vs VABS; center run for comparison) ==="]
rep.append("STRESS rms diff vs VABS (MPa):        WEB      SKIN")
for tag, s in (("center (production run)", sC), ("OML (this run)", sM)):
    rep.append("  %-24s  s11 %7.2f / %7.2f   s12 %7.2f / %7.2f"
               % (tag, rms(sVg[webp, 0] - s[webp, 0]), rms(sVg[~webp, 0] - s[~webp, 0]),
                  rms(sVg[webp, 5] - s[webp, 5]), rms(sVg[~webp, 5] - s[~webp, 5])))
rep.append("  VABS signal rms:          s11 %7.2f / %7.2f   s12 %7.2f / %7.2f"
           % (rms(sVg[webp, 0]), rms(sVg[~webp, 0]), rms(sVg[webp, 5]), rms(sVg[~webp, 5])))
rep.append("DISP (warping) rms diff vs VABS (mm), u1/u2/u3:")
for tag, u in (("center (production run)", uC), ("OML (this run)", uM)):
    rep.append("  %-24s  %7.4f  %7.4f  %7.4f" % (tag, rms(uV[:, 0] - u[:, 0]),
               rms(uV[:, 1] - u[:, 1]), rms(uV[:, 2] - u[:, 2])))
rep.append("Timo diag OML (MSG G): " + np.array2string(np.diag(C6), precision=4))
rep.append("timing: stress %.1fs, disp %.1fs" % (t_s, t_u))
print("\n".join(rep), flush=True)

np.savez(os.path.join(DATA, "msgrm_r020_oml_fields.npz"),
         sm_xy=sm_xy, stress_msgrm_oml=sM, stress_vabs=sVg,
         xy_nodes=xy, disp_msgrm_oml=uM, disp_vabs=uV,
         zoff_face=zoff_face, is_web=is_web, el_gauss=elg, z_gauss=zdg,
         C6_oml=C6, FF=FF)

# ================= CPB-style LINE figures at the archived path points =================
def beam_kinematics(path, node):
    Lf = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(Lf):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in Lf[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")])
            RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]])
            t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
            return u_g, C
    raise ValueError("no BeamDyn header")


u_g, Cbk = beam_kinematics(BD_VABS, 11)


def total_disp(u_warp_mm, node_xy):
    r3 = np.column_stack([np.zeros(len(node_xy)), node_xy[:, 0], node_xy[:, 1]])
    return (Cbk @ (u_warp_mm / 1e3 + r3).T).T + u_g - r3


def read_out(path):
    d = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        d[t[0]] = np.array([float(x) for x in t[1:]])
    return d


def plainaxis(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(alpha=0.3)


def rel_err(a, b):
    d = np.abs(a - b)
    keep = d <= 8.0 * np.median(d) + 1e-12
    return 100.0 * np.linalg.norm((a - b)[keep]) / (np.linalg.norm(b[keep]) + 1e-30)


tree_g = cKDTree(sm_xy)
tree_n = cKDTree(xy)
SCOMP = ["s_11", "s_12", "s_22"]
SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
SCOL = [0, 5, 1]
UCOMP = ["u_1", "u_2", "u_3"]
ULAB = [r"$u_1$", r"$u_2$", r"$u_3$"]
PATHS = {"iea_s10.circumferential": "dehom_r020_circ",
         "iea_s10.lp_sparcap_left_thickness": "dehom_r020_cap"}
for stem, fout in PATHS.items():
    vb = read_out(os.path.join(VBD, stem + ".out"))
    Pp = np.column_stack([vb["y2"], vb["y3"]])
    s = vb["non_dim_path"]
    s = (s - s.min()) / (s.max() - s.min() + 1e-30)
    dg, ig = tree_g.query(Pp)
    dn, iN = tree_n.query(Pp)
    sMp = sM[ig]
    uMp = total_disp(uM[iN], xy[iN])

    d0 = np.abs(sMp[:, 0] - vb["s_11"] / 1e6)
    keep = d0 <= 8.0 * np.median(d0) + 1e-12
    ndrop = int((~keep).sum())

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for k in range(3):
        ax = axs[k]
        ax.plot(s[keep], vb[SCOMP[k]][keep] / 1e6, "-o", color=VABSC, ms=3.5,
                lw=1.5, label="VABS")
        ax.plot(s[keep], sMp[keep, SCOL[k]], "--s", color=RMC, ms=3.5, mfc="none",
                mew=1.2, lw=1.4, label="OpenSG-RM (OML)")
        ax.set_ylabel("%s  [MPa]" % SLAB[k], fontsize=11)
        ax.set_xlabel("non-dimensional path coordinate", fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc="best")
    if ndrop:
        axs[0].text(0.03, 0.03, "%d ply-interface pt(s) hidden" % ndrop,
                    transform=axs[0].transAxes, va="bottom", fontsize=7, color="0.55")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fout + "_stress_oml.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for k in range(3):
        ax = axs[k]
        ax.plot(s, vb[UCOMP[k]], "-o", color=VABSC, ms=3.5, lw=1.5, label="VABS")
        ax.plot(s, uMp[:, k], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2, lw=1.4,
                label="OpenSG-RM (OML)")
        ax.set_ylabel("%s  [m]" % ULAB[k], fontsize=11)
        ax.set_xlabel("non-dimensional path coordinate", fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fout + "_disp_oml.png"), dpi=150)
    plt.close(fig)

    rep.append("%s (OML): stress " % stem + "  ".join("s%s %.1f%%" % (["11", "12", "22"][k],
               rel_err(sMp[:, SCOL[k]], vb[SCOMP[k]] / 1e6)) for k in range(3)))
    rep.append("  disp  : " + "  ".join("%s %.2f%%" % (["u1", "u2", "u3"][k],
               rel_err(uMp[:, k], vb[UCOMP[k]])) for k in range(3)))
    print("\n".join(rep[-2:]), flush=True)

# ================= contour ROW figures =================
Lf = [l for l in open(os.path.join(VABS, "iea_s10.sg")).read().splitlines() if l.strip()]
hi = next(i for i, l in enumerate(Lf) if len(l.split()) == 3
          and all(x.lstrip("-").isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
nn, ne, nm = [int(x) for x in Lf[hi].split()]
conn = [[int(x) for x in Lf[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    if len(c0) == 3:
        trl.append(c0[:3])
    elif len(c0) >= 4:
        trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]])
tris = np.array(trl)
M = tris.shape[0]
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


def row_fig(name, exploded, dataV, dataR, label, unit, rtitle):
    m = np.nanpercentile(np.abs(np.r_[dataV, dataR]), 99) or 1e-9
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 3.1))
    for a, dat, ttl in [(ax[0], dataV, "VABS"), (ax[1], dataR, rtitle)]:
        tr = etri if exploded else tri
        cs = a.tripcolor(tr, np.clip(dat, -m, m), shading="gouraud", cmap="rainbow",
                         vmin=-m, vmax=m)
        a.set_aspect("equal"); a.axis("off")
        a.set_title(ttl, fontsize=15, pad=4)
    cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.92, pad=0.012)
    cb.set_label("%s %s" % (label, unit), fontsize=15)
    cb.ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


for ci, lab, nmn in [(0, r"$\sigma_{11}$", "S11"), (1, r"$\sigma_{22}$", "S22"),
                     (5, r"$\sigma_{12}$", "S12")]:
    row_fig("r020_row_%s_oml.png" % nmn, True,
            corners_of(sVg[:, ci]), corners_of(sM[:, ci]), lab, "(MPa)", "OpenSG-RM (OML)")
for ci, lab, nmn in [(0, r"$u_1$", "u1"), (1, r"$u_2$", "u2"), (2, r"$u_3$", "u3")]:
    row_fig("r020_row_%s_oml.png" % nmn, False, uV[:, ci], uM[:, ci], lab, "(mm)",
            "OpenSG-RM (OML)")

open(os.path.join(DATA, "README_oml_metrics.txt"), "w").write("\n".join(rep) + "\n")
print("done", flush=True)
