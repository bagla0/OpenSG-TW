"""msgrm_r020_compare.py -- r=0.2 (iea_s10, mid-ref) Whitney-RM vs MSG-RM homo + dehom comparison.

(1) Emit the per-layup 8x8 wall-law yamls (ABDG storage), Whitney-G and MSG-G variants.
(2) HOMO: RM ring Timoshenko 6x6 with Whitney-G walls vs MSG-G walls (+ timing).
(3) DEHOM at all VABS .SM gauss points, three variants (+ timing):
      V1  baseline (cached)                      = shipped Whitney recovery
      V2  + WEB REGISTRATION fix                 (per-web-element wall-centre offset from the .sg)
      V3  + flow-avg rows(2,5) + MSG-RM step-2   (first-order gradient recovery, dE1 span + dE2 arc)
(4) sigma12 metrics web/skin + 4-panel contour (VABS | V1 | V2 | V3).
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
import yaml as _yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain, _flow_nodal_avg
from emit_abd import emit_station_abd, material_db_from_yaml
from msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from run_ring_indep import ring_indep
from oml_ring import load_ring_ref
from segment_indep import quad_ops_indep
from opensg_jax.fe_jax.msg_dehom import _project_point
from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, plate_stress_at_depth, rotation_6x6

SHELL = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]
FIG = os.path.join(HERE, "figures")
os.makedirs(os.path.join(HERE, "abd8"), exist_ok=True)

# ---------------- (1) 8x8 wall-law yamls ----------------
t0 = time.perf_counter()
outw = emit_station_abd(SHELL, os.path.join(HERE, "abd8", "iea_s10_abd8_whitney.yaml"),
                        station="iea_s10", r=0.20, ref="mid", g_source="whitney")
t_w8 = time.perf_counter() - t0
t0 = time.perf_counter()
outm = emit_station_abd(SHELL, os.path.join(HERE, "abd8", "iea_s10_abd8_msg.yaml"),
                        station="iea_s10", r=0.20, ref="mid", g_source="msg")
t_m8 = time.perf_counter() - t0
print("=== per-layup 2x2 G: Whitney vs MSG (mid-ref 8x8 emitted, %.2fs / %.2fs) ===" % (t_w8, t_m8))
Gw_by = {}; Gm_by = {}
for Lw, Lm in zip(outw["layups"], outm["layups"]):
    Gw = np.array(Lw["Gs"]); Gm = np.array(Lm["Gs"])
    Gw_by[Lw["name"]] = Gw; Gm_by[Lm["name"]] = Gm
    print(" %-10s G11 %.4e -> %.4e (%+6.1f%%)   G22 %.4e -> %.4e (%+6.1f%%)"
          % (Lw["name"], Gw[0, 0], Gm[0, 0], 100 * (Gm[0, 0] / Gw[0, 0] - 1),
             Gw[1, 1], Gm[1, 1], 100 * (Gm[1, 1] / Gw[1, 1] - 1)))

# ---------------- (2) HOMO: ring with Whitney vs MSG walls ----------------
d = _yaml.safe_load(open(SHELL))
sec_names = [s["elementSet"] for s in d["sections"]]
R = load_ring_ref(SHELL, "center")
args = (R["rx"], R["cells"], R["rsub"], R["re3"])
kw = dict(shear="mitc4_g23", lam_space="elem")

t0 = time.perf_counter()
C6w = ring_indep(*args, R["D_by"], R["G_by"], R["k22"], R["ax"], R["cross"], **kw)
t_hw = time.perf_counter() - t0
G_by_msg = [np.asarray(Gm_by[sec_names[si]]) for si in range(len(R["G_by"]))]
t0 = time.perf_counter()
C6m = ring_indep(*args, R["D_by"], G_by_msg, R["k22"], R["ax"], R["cross"], **kw)
t_hm = time.perf_counter() - t0
C6w = 0.5 * (np.asarray(C6w) + np.asarray(C6w).T); C6m = 0.5 * (np.asarray(C6m) + np.asarray(C6m).T)
lab = ["EA ", "GA2", "GA3", "GJ ", "EI2", "EI3"]
print("\n=== HOMO Timoshenko 6x6 diag: Whitney-G walls vs MSG-G walls (%.2fs / %.2fs) ===" % (t_hw, t_hm))
for i in range(6):
    print("  %s  %.6e   %.6e   %+8.4f %%" % (lab[i], C6w[i, i], C6m[i, i], 100 * (C6m[i, i] / C6w[i, i] - 1)))

# ---------------- (3) DEHOM ----------------
B = dehom_rm.build_rm_bundle(SHELL)
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
_, st_m6, st_cl1, st_cl2 = __import__("opensg_jax.fe_jax.msg_dehom", fromlist=["_macro_recovery"])._macro_recovery(np.asarray(B["Timo"]), np.linalg.inv(np.asarray(B["Timo"])) @ FF)

corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
nodes_s, quads_s, _hs = B["strip"]
n_el = rc.shape[0]
layups = B["layup_per_elem"]; ldb = B["layup_db"]; mdb = B["material_db"]
frac = float(B.get("frac", 0.0))

dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
P = len(sm_xy)
print("\nDEHOM: %d gauss points" % P, flush=True)

# --- project every gauss point once ---
t0 = time.perf_counter()
el = np.zeros(P, int); xia = np.zeros(P); zdep = np.zeros(P)
for ip in range(P):
    e, xi, pr = _project_point(corners, rc, sm_xy[ip])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    t2, t3 = corners[c1] - corners[c0]
    tl = float(np.hypot(t2, t3)); t2, t3 = t2 / tl, t3 / tl
    n2, n3 = t3, -t2
    if (cen[0] - pr[0]) * n2 + (cen[1] - pr[1]) * n3 < 0.0:
        n2, n3 = -n2, -n3
    el[ip] = e; xia[ip] = xi
    zdep[ip] = (sm_xy[ip, 0] - pr[0]) * n2 + (sm_xy[ip, 1] - pr[1]) * n3
t_proj = time.perf_counter() - t0
print("projection pass %.1fs" % t_proj, flush=True)

# --- web chains + per-web-element registration offset from the .sg gauss cloud ---
deg = np.zeros(int(rc.max()) + 1, int)
for a, b in rc:
    deg[a] += 1; deg[b] += 1
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
            arc = sum(np.linalg.norm(corners[rc[c][1]] - corners[rc[c][0]]) for c in chain)
            cv = corners[cur] - corners[j]; ch = np.linalg.norm(cv)
            if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                is_web[chain] = True
zoff = np.zeros(n_el)
for e in np.where(is_web)[0]:
    zz = zdep[el == e]
    if len(zz) >= 8:
        zoff[e] = 0.5 * (np.percentile(zz, 2) + np.percentile(zz, 98))
print("web elements: %d, |zoff| mean %.1f mm max %.1f mm"
      % (is_web.sum(), 1e3 * np.abs(zoff[is_web]).mean(), 1e3 * np.abs(zoff).max()), flush=True)

# --- per-element shell strains + gradients ---
s6mid = np.zeros((n_el, 6)); s2mid = np.zeros((n_el, 2)); dE1e = np.zeros((n_el, 6))
for e in range(n_el):
    s6, s2 = _rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    s6mid[e] = np.asarray(s6, float); s2mid[e] = np.asarray(s2, float)
    Xe = nodes_s[quads_s[e]]; e3e = B["re3"][e]
    BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]), B["cross"], B["ax"])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    dE1e[e] = np.asarray(BDe @ st_cl1 + BDh @ aB[g], float)          # spanwise d(s6)/dx1
# nodal-averaged s6 (all rows) for arc interpolation + arc gradient
n_nd = int(rc.max()) + 1
acc = np.zeros((n_nd, 6)); cnt = np.zeros(n_nd)
for e in range(n_el):
    for nd in (int(rc[e, 0]), int(rc[e, 1])):
        acc[nd] += s6mid[e]; cnt[nd] += 1
s6n = np.where((cnt == 2)[:, None], acc / np.maximum(cnt, 1)[:, None], np.nan)
dE2e = np.zeros((n_el, 6))
for e in range(n_el):
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    Le = float(np.linalg.norm(corners[c1] - corners[c0]))
    v0 = s6n[c0] if np.isfinite(s6n[c0, 0]) else s6mid[e]
    v1 = s6n[c1] if np.isfinite(s6n[c1, 0]) else s6mid[e]
    dE2e[e] = (v1 - v0) / Le

# --- plate warping objects per layup ---
warpW = {ln: compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mdb,
         n_per_layer=2, return_warping=True, elem_order=2)[2] for ln, i in ldb.items()}
t0 = time.perf_counter()
warpM = {ln: rm_plate_msg(i["thick"], i["angles"], i["mat_names"], mdb,
         fraction=frac) for ln, i in ldb.items()}
t_wm = time.perf_counter() - t0
print("MSG warp objects for %d layups: %.1fs" % (len(warpM), t_wm), flush=True)

hth = {ln: float(sum(i["thick"])) for ln, i in ldb.items()}


def run_variant(tag, use_zoff, use_msg):
    t0 = time.perf_counter()
    out = np.zeros((P, 6))
    for ip in range(P):
        e = el[ip]; xi = xia[ip]
        ln = layups[e]; h = hth[ln]
        z = zdep[ip] - (zoff[e] if use_zoff else 0.0)
        if use_msg:
            # flow-averaged rows 2,5 + linear arc interpolation of nodal s6
            c0, c1 = int(rc[e, 0]), int(rc[e, 1])
            s6 = s6mid[e].copy()
            for row in (2, 5):
                v0 = s6n[c0, row] if np.isfinite(s6n[c0, row]) else s6mid[e, row]
                v1 = s6n[c1, row] if np.isfinite(s6n[c1, row]) else s6mid[e, row]
                s6[row] = (1.0 - xi) * v0 + xi * v1
            Gam, Sig, ply = msgrm_strain_at_depth(warpM[ln], z, s6, dE1e[e], dE2e[e])
        else:
            s6, _s2 = _rm_shell_strain(B, e, xi, st_m, aA, aB)
            s6r = np.array(s6, float)
            z_oml = z + frac * h
            s6r[0:3] = s6r[0:3] - frac * h * s6r[3:6]
            Gam, Sig, ply = plate_stress_at_depth(warpW[ln], s6r, z_oml)
        Sig = rotation_6x6(-ply) @ np.asarray(Sig, float)
        out[ip] = Sig
    dt = time.perf_counter() - t0
    print("variant %s: %.1fs" % (tag, dt), flush=True)
    return out / 1e6, dt


sV1 = np.load(os.path.join(HERE, "_rm_s10_cache.npz"))["sRg"]
sV2, t2 = run_variant("V2 (+web registration)", True, False)
sV3, t3 = run_variant("V3 (+registration +MSG-RM recovery)", True, True)
np.savez(os.path.join(HERE, "_msgrm_variants.npz"), sV2=sV2, sV3=sV3, zoff=zoff, is_web=is_web)

# ---------------- (4) metrics + contour ----------------
webp = is_web[el]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


print("\n=== sigma12 rms diff vs VABS (MPa) ===")
print("%-38s %8s %8s" % ("variant", "WEB", "SKIN"))
for tag, s in (("V1 baseline (shipped)", sV1), ("V2 +web registration", sV2),
               ("V3 +registration +MSG-RM recovery", sV3)):
    print("%-38s %8.3f %8.3f" % (tag, rms(sVg[webp, 5] - s[webp, 5]), rms(sVg[~webp, 5] - s[~webp, 5])))
print("VABS rms sigma12: web %.3f skin %.3f" % (rms(sVg[webp, 5]), rms(sVg[~webp, 5])))
print("\n=== sigma11 rms diff vs VABS (MPa) ===")
for tag, s in (("V1", sV1), ("V2", sV2), ("V3", sV3)):
    print("%-38s %8.3f %8.3f" % (tag, rms(sVg[webp, 0] - s[webp, 0]), rms(sVg[~webp, 0] - s[~webp, 0])))

# contour: sigma12 row, 4 panels, exploded gauss->corner (same recipe as r020_dehom_s10)
U = np.loadtxt(os.path.join(VABS, "iea_s10.sg.U")); U = U[np.argsort(U[:, 0])]
xy = U[:, 1:3]


def parse_sg_conn(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    hi = next(i for i, l in enumerate(L) if len(l.split()) == 3
              and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne, nm = [int(x) for x in L[hi].split()]
    return [[int(x) for x in L[hi + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]


conn = parse_sg_conn(os.path.join(VABS, "iea_s10.sg"))
trl = []
for c in conn:
    c0 = [n - 1 for n in c]
    if len(c0) == 3:
        trl.append(c0[:3])
    elif len(c0) >= 4:
        trl.extend([[c0[0], c0[1], c0[2]], [c0[0], c0[2], c0[3]]])
tris = np.array(trl); M = tris.shape[0]
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
eg = tri.get_trifinder()(sm_xy[:, 0], sm_xy[:, 1])
ordered = eg.size == 3 * M and np.array_equal(eg.reshape(M, 3), np.arange(M)[:, None].repeat(3, 1))
gxy = sm_xy.reshape(M, 3, 2)
Ai = np.linalg.inv(np.concatenate([np.ones((M, 3, 1)), gxy], 2))
Cc = np.concatenate([np.ones((M, 3, 1)), xy[tris]], 2)
EP = xy[tris].reshape(-1, 2)
etri = mtri.Triangulation(EP[:, 0], EP[:, 1], np.arange(3 * M).reshape(M, 3))
fig, ax = plt.subplots(1, 4, figsize=(26, 5.4))
panels = [("VABS", sVg[:, 5]), ("RM baseline", sV1[:, 5]),
          ("RM +web reg.", sV2[:, 5]), ("RM +reg +MSG-RM", sV3[:, 5])]
mlim = np.nanpercentile(np.abs(np.r_[sVg[:, 5], sV2[:, 5]]), 99) or 1e-9
for k, (tt, val) in enumerate(panels):
    g = val.reshape(M, 3, 1)
    corner = (Cc @ (Ai @ g))
    corner = np.clip(corner, g.min(1, keepdims=True), g.max(1, keepdims=True)).reshape(-1)
    cs = ax[k].tripcolor(etri, np.clip(corner, -mlim, mlim), shading="gouraud", cmap="rainbow",
                         vmin=-mlim, vmax=mlim)
    ax[k].set_aspect("equal"); ax[k].axis("off"); ax[k].set_title(tt, fontsize=18)
cb = fig.colorbar(cs, ax=ax.tolist(), shrink=0.8, pad=0.01)
cb.set_label("MPa", fontsize=16)
fig.savefig(os.path.join(FIG, "r020_s12_variants.png"), dpi=140, bbox_inches="tight")
print("wrote figures/r020_s12_variants.png")
print("\nTIMING: homo %.2f/%.2f s | 8x8 emit %.1f/%.1f s | proj %.0f s | V2 %.0f s | V3 %.0f s"
      % (t_hw, t_hm, t_w8, t_m8, t_proj, t2, t3))
