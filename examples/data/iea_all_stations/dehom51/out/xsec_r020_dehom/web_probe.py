"""web_probe.py -- why is the RM web sigma12 wrong at r=0.2?

(1) Per web chain: RM shell-strain rows 2 (2e12) and 5 (2k12) at element mids,
    split A=macro(BDe@st_m) / B=contour-deriv(BDh@wA) / C=axial-deriv(BDl@wB);
    physical expectation: B+C redistributes the uniform macro projection into
    the true shear flow.  Prints how much cancellation actually happens.
(2) Per web chain: profile along web height -- VABS sigma12 (gauss, from .SM)
    vs cached RM sigma12 at the SAME points, binned.
(3) Through-thickness sigma12 at mid-height of each web (VABS vs RM).
(4) Section macro state st = C6^{-1} FF printed (gamma2, gamma3, kappa1 scale).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
XSEC = os.path.abspath(os.path.join(ROOT, "..", "..", "TW-paper", "xsec_paper"))
sys.path.insert(0, XSEC)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import jax

jax.config.update("jax_enable_x64", True)
import dehom_rm
from dehom_rm import _macro_fields, _rm_shell_strain

SHELL = os.path.join(ROOT, "shell51", "1d_yaml", "iea_s10_shell.yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
FF = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))[10, 1:]

B = dehom_rm.build_rm_bundle(SHELL)
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
print("st = C6^-1 FF  [e1, g2, g3, k1, k2, k3]:", np.array2string(np.asarray(st), precision=4))
C6 = np.asarray(B["Timo"])
print("diag C6:", np.array2string(np.diag(C6), precision=3))

corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
deg = np.zeros(int(rc.max()) + 1, int)
for a, b in rc:
    deg[a] += 1; deg[b] += 1
adj = {}
for e, (a, b) in enumerate(rc):
    adj.setdefault(a, []).append((b, e))
    adj.setdefault(b, []).append((a, e))
junc = set(np.where(deg >= 3)[0])
chains, seen = [], set()
for j in junc:
    for (nxt, e0) in adj[j]:
        if e0 in seen:
            continue
        chain, nodes_c, prev, cur = [e0], [j], j, nxt
        seen.add(e0)
        while cur not in junc and deg[cur] == 2:
            (n1, e1), (n2_, e2) = adj[cur][0], adj[cur][1]
            nn, ee = (n1, e1) if n1 != prev else (n2_, e2)
            if ee in seen:
                break
            chain.append(ee); seen.add(ee)
            nodes_c.append(cur); prev, cur = cur, nn
        nodes_c.append(cur)
        if cur in junc:
            arc = sum(np.linalg.norm(corners[rc[c][1]] - corners[rc[c][0]]) for c in chain)
            cv = corners[cur] - corners[j]
            ch = np.linalg.norm(cv)
            if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                chains.append((chain, j, cur))
print("web chains found:", len(chains), " lengths:", [len(c[0]) for c in chains])

layups = B["layup_per_elem"]

for wi, (chain, j0, j1) in enumerate(sorted(chains, key=lambda c: corners[rc[c[0][0]][0]][0])):
    xw = np.mean([corners[rc[e]].mean(0)[0] for e in chain])
    print("\n================ WEB %d  (x2 ~ %.3f m, %d elements, layup %s) ================"
          % (wi, xw, len(chain), layups[chain[len(chain) // 2]]))
    print(" %-4s %-8s | %-36s | %-36s" % ("k", "y3 (m)", "row2 2e12: A_macro B_cont C_axial tot", "row5 2k12: A  B  C  tot"))
    A2s = np.zeros(len(chain)); B2s = np.zeros(len(chain)); C2s = np.zeros(len(chain))
    for k, e in enumerate(chain):
        nodes, quads, _h = B["strip"]
        Xe = nodes[quads[e]]; e3e = B["re3"][e]
        from segment_indep import quad_ops_indep
        BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B["k22"][e]),
                                           B["cross"], B["ax"])
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
        A2 = float((BDe @ st_m)[2]); B2 = float((BDh @ aA[g])[2]); C2 = float((BDl @ aB[g])[2])
        A5 = float((BDe @ st_m)[5]); B5 = float((BDh @ aA[g])[5]); C5 = float((BDl @ aB[g])[5])
        y3m = corners[rc[e]].mean(0)[1]
        A2s[k], B2s[k], C2s[k] = A2, B2, C2
        if k % max(1, len(chain) // 10) == 0 or k == len(chain) - 1:
            print(" %-4d %-8.3f | %+9.2e %+9.2e %+9.2e %+9.2e | %+9.2e %+9.2e %+9.2e %+9.2e"
                  % (k, y3m, A2, B2, C2, A2 + B2 + C2, A5, B5, C5, A5 + B5 + C5))
    tot = A2s + B2s + C2s
    print(" chain means: A %.3e  B %.3e  C %.3e  tot %.3e   cancel ratio (B+C)/A = %.2f"
          % (A2s.mean(), B2s.mean(), C2s.mean(), tot.mean(), -(B2s.mean() + C2s.mean()) / (A2s.mean() + 1e-30)))

# ---- VABS vs cached RM sigma12 along each web (bin by y3) + through thickness ----
dsm = np.loadtxt(os.path.join(VABS, "iea_s10.sg.SM"), skiprows=2)
sm_xy = dsm[:, :2]
sVg = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
sRg = np.load(os.path.join(HERE, "_rm_s10_cache.npz"))["sRg"]

for wi, (chain, j0, j1) in enumerate(sorted(chains, key=lambda c: corners[rc[c[0][0]][0]][0])):
    xw = np.mean([corners[rc[e]].mean(0)[0] for e in chain])
    y3lo = min(corners[rc[e]].mean(0)[1] for e in chain)
    y3hi = max(corners[rc[e]].mean(0)[1] for e in chain)
    m = (np.abs(sm_xy[:, 0] - xw) < 0.12) & (sm_xy[:, 1] > y3lo + 0.03) & (sm_xy[:, 1] < y3hi - 0.03)
    print("\n---- WEB %d gauss profile (%d pts): y3-bin | VABS s12 mean(rms) | RM s12 mean(rms) MPa" % (wi, m.sum()))
    yb = np.linspace(y3lo, y3hi, 11)
    for bb in range(10):
        mm = m & (sm_xy[:, 1] >= yb[bb]) & (sm_xy[:, 1] < yb[bb + 1])
        if mm.sum() == 0:
            continue
        print("   %+.2f..%+.2f | %+8.2f (%7.2f) | %+8.2f (%7.2f)   n=%d"
              % (yb[bb], yb[bb + 1], sVg[mm, 5].mean(), np.sqrt((sVg[mm, 5] ** 2).mean()),
                 sRg[mm, 5].mean(), np.sqrt((sRg[mm, 5] ** 2).mean()), mm.sum()))
    ymid = 0.5 * (y3lo + y3hi)
    mm = m & (np.abs(sm_xy[:, 1] - ymid) < 0.06)
    if mm.sum() > 4:
        xs = sm_xy[mm, 0]
        order = np.argsort(xs)
        print("   through-thickness at mid-height (x2 offset from web line, VABS s12, RM s12):")
        for i in order[:: max(1, len(order) // 12)]:
            idx = np.where(mm)[0][i]
            print("     %+7.4f  %+8.2f  %+8.2f" % (sm_xy[idx, 0] - xw, sVg[idx, 5], sRg[idx, 5]))
