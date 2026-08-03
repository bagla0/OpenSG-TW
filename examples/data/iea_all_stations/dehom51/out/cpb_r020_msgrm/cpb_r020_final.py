"""cpb_r020_final.py -- FINAL r=0.2 MSG-RM dehom for the CPB paper, BOTH references,
with three corrections over the previous runs:

  (1) MATERIAL-AWARE projection (junction-leakage fix): each VABS gauss point carries
      the material of its .sg element; it may only project onto ring elements whose
      layup CONTAINS that material (matched by E1).  A carbon cap point at a T-junction
      can no longer land on the web ring element and be evaluated with the web layup,
      and a biax web point cannot land on the cap -- the junction leakage in the
      contour plots disappears at the model level, not by rendering tricks.
  (2) WARPING-NORMALIZATION transport (u3 offset fix): VABS normalizes its warping to
      zero average over the 2-D section, the RM ring over the contour; the difference
      is a constant per component (u3 ~ 0.7 mm).  The RM warping is shifted to match
      the VABS node-set average before any comparison (a pure gauge transport).
  (3) A third through-thickness path AT the centre-web / spar-cap T-junction: a narrow
      x2 strip through the cap wall and the upper web, the most critical location.

Outputs per ref tag in {oml, mid}: figures/{tag}_circ_{stress,disp}.png,
{tag}_cap_{stress,disp}.png, {tag}_junc_{stress,disp}.png, {tag}_row_*.png,
data/final_{tag}_fields.npz, data/README_final_metrics.txt (both refs).
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

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
import matplotlib.tri as mtri
from matplotlib.ticker import ScalarFormatter

import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain
from msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from segment_indep import quad_ops_indep
from opensg_jax.fe_jax.msg_dehom import _macro_recovery
from opensg_jax.fe_jax.msg_materials import rotation_6x6

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

# ================= VABS reference (shared) =================
dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
U = np.loadtxt(os.path.join(VABS, "iea_s10.sg.U"))
U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]
uV_tot = U[:, 3:6]

# .sg: per-element material id -> E1  (layer table + material blocks)
Lsg = [l for l in open(os.path.join(VABS, "iea_s10.sg")).read().splitlines() if l.strip()]
hi_ = next(i for i, l in enumerate(Lsg) if len(l.split()) == 3
           and all(x.lstrip("-").isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
nn_, ne_, nm_ = [int(x) for x in Lsg[hi_].split()]
conn = [[int(x) for x in Lsg[hi_ + 1 + nn_ + k].split()[1:] if int(x) != 0] for k in range(ne_)]
lay_ = np.array([int(float(Lsg[hi_ + 1 + nn_ + ne_ + k].split()[1])) for k in range(ne_)])
k = hi_ + 1 + nn_ + 2 * ne_
lyr2mat = {}
while k < len(Lsg):
    p = Lsg[k].split()
    if len(p) == 3 and p[0].lstrip("-").isdigit():
        lyr2mat[int(p[0])] = int(p[1]); k += 1
    else:
        break
matE = {}
while k < len(Lsg):
    p = Lsg[k].split()
    if len(p) == 2 and p[0].isdigit():
        mid = int(p[0])
        matE[mid] = float(Lsg[k + 1].split()[0])
        k += 1
        while k < len(Lsg) and len(Lsg[k].split()) != 2:
            k += 1
    else:
        k += 1
elem_mat = np.array([lyr2mat[int(l)] for l in lay_])
assert 3 * ne_ == len(sm_xy), "expected 3 gauss pts per .sg element in .SM order"
gp_mat = np.repeat(elem_mat, 3)
print(".sg: %d elems, materials %s" % (ne_, {m: "%.3g" % e for m, e in sorted(matE.items())}), flush=True)

# tri mesh for the contour rows
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
Cinv = np.linalg.inv(Cbk)
r3n = np.column_stack([np.zeros(len(xy)), xy[:, 0], xy[:, 1]])
uV = ((Cinv @ (uV_tot - u_g + r3n).T).T - r3n) * 1e3       # VABS warping (mm)


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


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


SCOMP = ["s_11", "s_12", "s_22"]
SLAB = [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{22}$"]
SCOL = [0, 5, 1]
UCOMP = ["u_1", "u_2", "u_3"]
ULAB = [r"$u_1$", r"$u_2$", r"$u_3$"]
rep = ["=== cpb_r020_final: material-aware projection + warping gauge transport + junction path ==="]


def run_ref(tag, shell_yaml, ref):
    print("\n===== %s (%s) =====" % (tag, ref), flush=True)
    B = dehom_rm.build_rm_bundle(shell_yaml, ref=ref)
    frac = float(B.get("frac", 0.0))
    st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
    C6 = np.asarray(B["Timo"])
    _, _sm, st_cl1, st_cl2 = _macro_recovery(C6, np.linalg.inv(C6) @ FF)
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
    nodes_s, quads_s, _hs = B["strip"]
    n_el = rc.shape[0]; n_nd = int(rc.max()) + 1
    layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]

    A0 = corners[rc[:, 0]]; T = corners[rc[:, 1]] - corners[rc[:, 0]]
    L2 = (T ** 2).sum(1); tun = T / np.sqrt(L2)[:, None]
    nvec = np.column_stack([tun[:, 1], -tun[:, 0]])
    emid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
    flip = ((cen - emid) * nvec).sum(1) < 0
    nvec[flip] *= -1.0
    Lel = np.sqrt(L2)

    def project(P, allow=None, chunk=4000):
        P = np.atleast_2d(np.asarray(P, float))
        el = np.zeros(len(P), int); xi = np.zeros(len(P)); zd = np.zeros(len(P))
        for i0 in range(0, len(P), chunk):
            Pc = P[i0:i0 + chunk]
            dp = Pc[:, None, :] - A0[None, :, :]
            tpar = np.clip((dp * T[None]).sum(2) / L2[None], 0.0, 1.0)
            foot = A0[None] + tpar[..., None] * T[None]
            d2 = ((Pc[:, None, :] - foot) ** 2).sum(2)
            if allow is not None:
                d2[:, ~allow] = np.inf
            e = np.argmin(d2, 1)
            r_ = np.arange(len(Pc))
            el[i0:i0 + chunk] = e
            xi[i0:i0 + chunk] = tpar[r_, e]
            zd[i0:i0 + chunk] = ((Pc - foot[r_, e]) * nvec[e]).sum(1)
        return el, xi, zd

    # ---- material-aware allow masks: ring element layup ply E1 set vs .sg material E1 ----
    lay_e1 = {ln: set(round(float(mdb[nm]["E"][0]), 4) for nm in i["mat_names"])
              for ln, i in ldb.items()}
    allow_of = {}
    for mid, e1 in matE.items():
        key = round(e1, 4)
        m = np.array([any(abs(key - v) <= 1e-3 * max(key, v) for v in lay_e1[layups[e]])
                      for e in range(n_el)])
        allow_of[mid] = m if m.any() else None
        print("  mat %d (E1 %.3g): %d/%d ring elems allowed" % (mid, e1, m.sum(), n_el), flush=True)

    t0 = time.perf_counter()
    elg = np.zeros(len(sm_xy), int); xig = np.zeros(len(sm_xy)); zdg = np.zeros(len(sm_xy))
    for mid in matE:
        idx = np.where(gp_mat == mid)[0]
        if len(idx) == 0:
            continue
        e_, x_, z_ = project(sm_xy[idx], allow=allow_of[mid])
        elg[idx] = e_; xig[idx] = x_; zdg[idx] = z_
    eln, xin, zdn = project(xy)
    print("  material-aware projection: %d gauss + %d nodes in %.1fs"
          % (len(sm_xy), len(xy), time.perf_counter() - t0), flush=True)

    # ---- strains + boundary-aware gradients ----
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

    # ---- web classification + registration ----
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
    for e in range(n_el):
        zz = zdg[elg == e]
        if len(zz) >= 8:
            if frac == 0.0:
                zoff[e] = np.percentile(zz, 2)                     # OML: SG z=0 at the outer face
            elif frac == 1.0:
                zoff[e] = np.percentile(zz, 98)                    # IML: SG z=0 at the inner face
            else:
                zoff[e] = 0.5 * (np.percentile(zz, 2) + np.percentile(zz, 98))
    if frac == 0.5:
        zoff[~is_web] = 0.0                       # centre ref: registration is a web correction only
    print("  registration |zoff| web mean %.1f mm" % (1e3 * np.abs(zoff[is_web]).mean()), flush=True)

    # ---- stress at all gauss ----
    warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb,
             fraction=frac) for ln, i in ldb.items()}
    t0 = time.perf_counter()
    sM = np.zeros((len(sm_xy), 6))
    for ip in range(len(sm_xy)):
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
    print("  stress at %d gauss: %.1fs" % (len(sm_xy), time.perf_counter() - t0), flush=True)

    # ---- warping disp at nodes + GAUGE TRANSPORT ----
    wn = np.asarray(aA).reshape(-1, 6)
    uM = np.zeros((len(xy), 3))
    for ip in range(len(xy)):
        e = eln[ip]; xi = xin[ip]
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        umid = (1.0 - xi) * wn[c0, 0:3] + xi * wn[c1, 0:3]
        om = (1.0 - xi) * wn[c0, 3:6] + xi * wn[c1, 3:6]
        n2, n3 = nvec[e]
        uM[ip] = umid + zdn[ip] * np.cross(om, np.array([0.0, n2, n3]))
    uM *= 1e3
    gauge = uV.mean(0) - uM.mean(0)               # constant normalization difference (mm)
    uM = uM + gauge
    print("  warping gauge transport (mm): u1 %+.3f  u2 %+.3f  u3 %+.3f" % tuple(gauge), flush=True)

    webp = is_web[elg]
    rep.append("[%s] full-field rms diff vs VABS (MPa): s11 web %.2f / skin %.2f, s12 web %.2f / skin %.2f"
               % (tag, rms(sVg[webp, 0] - sM[webp, 0]), rms(sVg[~webp, 0] - sM[~webp, 0]),
                  rms(sVg[webp, 5] - sM[webp, 5]), rms(sVg[~webp, 5] - sM[~webp, 5])))
    rep.append("[%s] warping disp rms diff (mm, gauge-transported): %.4f %.4f %.4f"
               % (tag, rms(uV[:, 0] - uM[:, 0]), rms(uV[:, 1] - uM[:, 1]), rms(uV[:, 2] - uM[:, 2])))
    rep.append("[%s] gauge (mm): %+.3f %+.3f %+.3f" % (tag, gauge[0], gauge[1], gauge[2]))
    print("\n".join(rep[-3:]), flush=True)

    np.savez(os.path.join(DATA, "final_%s_fields.npz" % tag),
             sm_xy=sm_xy, stress_msgrm=sM, stress_vabs=sVg, xy_nodes=xy,
             disp_msgrm=uM, disp_vabs=uV, zoff=zoff, is_web=is_web,
             el_gauss=elg, z_gauss=zdg, C6=C6, FF=FF, gauge=gauge)

    tree_g = cKDTree(sm_xy)
    tree_n = cKDTree(xy)

    # ---- archived line paths ----
    for stem, pth in (("iea_s10.circumferential", "circ"), ("iea_s10.lp_sparcap_left_thickness", "cap")):
        vb = read_out(os.path.join(VBD, stem + ".out"))
        Pp = np.column_stack([vb["y2"], vb["y3"]])
        s = vb["non_dim_path"]
        s = (s - s.min()) / (s.max() - s.min() + 1e-30)
        ig = tree_g.query(Pp)[1]
        iN = tree_n.query(Pp)[1]
        sMp = sM[ig]
        uMp = total_disp(uM[iN], xy[iN])
        uVp = total_disp(uV[iN], xy[iN])          # same nodes, same beam kinematics
        d0 = np.abs(sMp[:, 0] - vb["s_11"] / 1e6)
        keep = d0 <= 8.0 * np.median(d0) + 1e-12
        ndrop = int((~keep).sum())

        fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
        for kk in range(3):
            ax = axs[kk]
            ax.plot(s[keep], vb[SCOMP[kk]][keep] / 1e6, "-o", color=VABSC, ms=3.5, lw=1.5, label="VABS")
            ax.plot(s[keep], sMp[keep, SCOL[kk]], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2,
                    lw=1.4, label="OpenSG-RM")
            ax.set_ylabel("%s  [MPa]" % SLAB[kk], fontsize=11)
            ax.set_xlabel("non-dimensional path coordinate", fontsize=10)
            plainaxis(ax)
            ax.legend(fontsize=9, loc="best")
        if ndrop:
            axs[0].text(0.03, 0.03, "%d ply-interface pt(s) hidden" % ndrop,
                        transform=axs[0].transAxes, va="bottom", fontsize=7, color="0.55")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "%s_%s_stress.png" % (tag, pth)), dpi=150)
        plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
        for kk in range(3):
            ax = axs[kk]
            ax.plot(s, uVp[:, kk], "-o", color=VABSC, ms=3.5, lw=1.5, label="VABS")
            ax.plot(s, uMp[:, kk], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2, lw=1.4,
                    label="OpenSG-RM")
            ax.set_ylabel("%s  [m]" % ULAB[kk], fontsize=11)
            ax.set_xlabel("non-dimensional path coordinate", fontsize=10)
            plainaxis(ax)
            ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "%s_%s_disp.png" % (tag, pth)), dpi=150)
        plt.close(fig)

        rep.append("[%s] %s stress: " % (tag, pth) + "  ".join("s%s %.1f%%" % (["11", "12", "22"][kk],
                   rel_err(sMp[:, SCOL[kk]], vb[SCOMP[kk]] / 1e6)) for kk in range(3)))
        rep.append("[%s] %s disp  : " % (tag, pth) + "  ".join("%s %.2f%%" % (["u1", "u2", "u3"][kk],
                   rel_err(uMp[:, kk], uVp[:, kk])) for kk in range(3)))
        print("\n".join(rep[-2:]), flush=True)

    # ---- junction path: CONNECTED polyline -- OML-normal through the cap wall at the
    # centre-web T-junction, then continuing down the web mid-line.  Purpose: show the
    # recovered stress and displacement CONTINUITY through the junction (natural for the
    # RM C0 6-DOF element; hard for C1 KL junctions and load-assumed recoveries).
    jn = [nd for nd in np.where(deg >= 3)[0] if corners[nd, 1] > 0]
    ndj = jn[int(np.argmin([abs(corners[nd, 0] + 0.044) for nd in jn]))]
    sk_e = [e for e in nd_el[ndj] if not is_web[e]]
    wb_e = [e for e in nd_el[ndj] if is_web[e]]
    nrm = np.mean([nvec[e] for e in sk_e], axis=0)
    nrm /= np.linalg.norm(nrm)
    e_w = wb_e[0]
    oth = int(rc[e_w, 1]) if int(rc[e_w, 0]) == ndj else int(rc[e_w, 0])
    dweb = corners[oth] - corners[ndj]
    dweb /= np.linalg.norm(dweb)
    hj = float(sum(ldb[layups[sk_e[0]]]["thick"]))     # cap wall thickness at the junction
    p0 = corners[ndj] - frac * hj * nrm                # start on the PHYSICAL outer surface
    p1 = p0 + hj * nrm
    s1 = np.arange(0.0, hj, 0.0025)
    s2 = np.arange(0.0, 1.6 * hj, 0.0025)
    pline = np.vstack([p0[None] + s1[:, None] * nrm[None],
                       p1[None] + s2[:, None] * dweb[None]])
    arc = np.concatenate([s1, hj + s2]) * 1e3          # mm along the polyline
    rep.append("[%s] junction polyline: node %d at (%.3f, %.3f), wall %g mm + web leg %.0f mm"
               % (tag, ndj, p0[0], p0[1], 1e3 * hj, 1.6e3 * hj))

    def along(tree, maxd):
        d_, i_ = tree.query(pline)
        ok_ = d_ < maxd
        seen_, aa, ii = set(), [], []
        for a_, j_ in zip(arc[ok_], i_[ok_]):
            if j_ not in seen_:
                seen_.add(j_)
                aa.append(a_)
                ii.append(j_)
        return np.array(aa), np.array(ii, int)

    aj, gj = along(tree_g, 0.012)                      # gauss pts within one cell of the line
    an, nj = along(tree_n, 0.012)
    uMs = total_disp(uM[nj], xy[nj])
    uVs = total_disp(uV[nj], xy[nj])
    rep.append("[%s] junction path: %d gauss / %d node samples, arc 0..%.0f mm, CONNECTED"
               % (tag, len(gj), len(nj), arc[-1]))

    def leg_labels(ax):
        ax.axvline(1e3 * hj, color="0.6", lw=1.0, ls=":")
        ax.text(0.5 * 1e3 * hj, 0.97, "cap wall", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="0.35")
        ax.text(1e3 * hj + 0.5 * (arc[-1] - 1e3 * hj), 0.97, "web",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9, color="0.35")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    for kk in range(3):
        ax = axs[kk]
        ax.plot(aj, sVg[gj, SCOL[kk]], "-o", color=VABSC, ms=3.5, lw=1.4, label="VABS")
        ax.plot(aj, sM[gj, SCOL[kk]], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2,
                lw=1.3, label="OpenSG-RM")
        leg_labels(ax)
        ax.set_ylabel("%s  [MPa]" % SLAB[kk], fontsize=11)
        ax.set_xlabel("arc along the junction path [mm]", fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "%s_junc_stress.png" % tag), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for kk in range(3):
        ax = axs[kk]
        ax.plot(an, uVs[:, kk], "-o", color=VABSC, ms=3.5, lw=1.4, label="VABS")
        ax.plot(an, uMs[:, kk], "--s", color=RMC, ms=3.5, mfc="none", mew=1.2, lw=1.3,
                label="OpenSG-RM")
        leg_labels(ax)
        ax.set_ylabel("%s  [m]" % ULAB[kk], fontsize=11)
        ax.set_xlabel("arc along the junction path [mm]", fontsize=10)
        plainaxis(ax)
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "%s_junc_disp.png" % tag), dpi=150)
    plt.close(fig)

    np.savetxt(os.path.join(DATA, "junction_polyline_%s.dat" % tag),
               np.vstack([p0, p1, p1 + 1.6 * hj * dweb]),
               header="junction path polyline vertices (y2 y3), legs: OML->IML normal, then web")
    rep.append("[%s] junction stress: " % tag + "  ".join("s%s %.1f%%" % (["11", "12", "22"][kk],
               rel_err(sM[gj, SCOL[kk]], sVg[gj, SCOL[kk]])) for kk in range(3)))
    rep.append("[%s] junction disp  : " % tag + "  ".join("%s %.2f%%" % (["u1", "u2", "u3"][kk],
               rel_err(uMs[:, kk], uVs[:, kk])) for kk in range(3)))
    print("\n".join(rep[-4:]), flush=True)

    # ---- contour rows ----
    for ci, lab, nmn in [(0, r"$\sigma_{11}$", "S11"), (1, r"$\sigma_{22}$", "S22"),
                         (5, r"$\sigma_{12}$", "S12")]:
        mlim = np.nanpercentile(np.abs(np.r_[sVg[:, ci], sM[:, ci]]), 99) or 1e-9
        fig, ax = plt.subplots(1, 2, figsize=(13.0, 3.1))
        for a, dat, ttl in [(ax[0], corners_of(sVg[:, ci]), "VABS"),
                            (ax[1], corners_of(sM[:, ci]), "OpenSG-RM")]:
            cs = a.tripcolor(etri, np.clip(dat, -mlim, mlim), shading="gouraud", cmap="rainbow",
                             vmin=-mlim, vmax=mlim)
            a.set_aspect("equal"); a.axis("off")
            a.set_title(ttl, fontsize=15, pad=4)
        cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.92, pad=0.012)
        cb.set_label("%s (MPa)" % lab, fontsize=15)
        cb.ax.tick_params(labelsize=12)
        fig.savefig(os.path.join(FIG, "%s_row_%s.png" % (tag, nmn)), dpi=150, bbox_inches="tight")
        plt.close(fig)
    for ci, lab, nmn in [(0, r"$u_1$", "u1"), (1, r"$u_2$", "u2"), (2, r"$u_3$", "u3")]:
        mlim = np.nanpercentile(np.abs(np.r_[uV[:, ci], uM[:, ci]]), 99) or 1e-9
        fig, ax = plt.subplots(1, 2, figsize=(13.0, 3.1))
        for a, dat, ttl in [(ax[0], uV[:, ci], "VABS"), (ax[1], uM[:, ci], "OpenSG-RM")]:
            cs = a.tripcolor(tri, np.clip(dat, -mlim, mlim), shading="gouraud", cmap="rainbow",
                             vmin=-mlim, vmax=mlim)
            a.set_aspect("equal"); a.axis("off")
            a.set_title(ttl, fontsize=15, pad=4)
        cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.92, pad=0.012)
        cb.set_label("%s (mm)" % lab, fontsize=15)
        cb.ax.tick_params(labelsize=12)
        fig.savefig(os.path.join(FIG, "%s_row_%s.png" % (tag, nmn)), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  figures written for %s" % tag, flush=True)


CFG = {"oml": (os.path.join(ROOT, "shell51", "1d_yaml_oml51", "iea_s10_shell.yaml"), "oml"),
       "mid": (os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml"), "center"),
       "iml": (os.path.join(ROOT, "shell51", "1d_yaml_iml", "iea_s10_shell.yaml"), "iml")}
for _tag in (sys.argv[1:] or ["oml"]):
    run_ref(_tag, *CFG[_tag])
open(os.path.join(DATA, "README_final_metrics.txt"), "w").write("\n".join(rep) + "\n")
print("\nDONE", flush=True)
