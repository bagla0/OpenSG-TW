'''debug_s12_oml.py -- decompose the circumferential sigma12 sawtooth by term.

For every element of the OML ring (loop-ordered), evaluate the recovered shell
strain row 2eps12 (s6[2]) at the element midpoint, split into its three sources:
    A = (BDe @ st_m)[2]   -- macro/geometry term  (Rn*kappa1 etc., smooth)
    B = (BDh @ wA)[2]     -- CONTOUR DERIVATIVE of the warping (piecewise const)
    C = (BDl @ wB)[2]     -- axial-derivative (shear-warping) term
and likewise the twist row s6[5].  Prints jump statistics per term and writes a
diagnostic figure.  This identifies the oscillating term before any fix.
'''
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, '..'))
XS = next(c for c in [os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper'),
                      r'Y:\OpenSG-TW-claude\examples\TW-paper\xsec_paper'] if os.path.isdir(c))
REPO = os.path.abspath(os.path.join(XS, '..', '..', '..'))
for q in (XS, REPO, os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import jax

jax.config.update('jax_enable_x64', True)
import dehom_rm
from dehom_rm import _macro_fields
from segment_indep import quad_ops_indep

SHELL = os.path.join(IEA, 'shell51', '1d_yaml_oml', 'iea_s10_shell.yaml')
FF = np.loadtxt(os.path.join(HERE, 'beamdyn', 'ff51_rmc_reform.dat'))[10, 1:]

B = dehom_rm.build_rm_bundle(SHELL)
st, st_m, aA, aB = _macro_fields(B, beam_force_vabs=FF)
corners = np.asarray(B['corners']); rc = np.asarray(B['red_cells'])
nodes, quads, _h = B['strip']

# skin loop order (exclude webs) by angle about the centroid
deg = np.zeros(len(corners), int)
for a, b in rc:
    deg[a] += 1
    deg[b] += 1
mid = 0.5 * (corners[rc[:, 0]] + corners[rc[:, 1]])
cen = corners.mean(0)
# web detection identical to the yaml builder
adj = {}
for e, (a, b) in enumerate(rc):
    adj.setdefault(a, []).append((b, e))
    adj.setdefault(b, []).append((a, e))
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
            (n1, e1), (n2_, e2) = adj[cur][0], adj[cur][1]
            nn, ee = (n1, e1) if n1 != prev else (n2_, e2)
            if ee in seen:
                break
            chain.append(ee)
            seen.add(ee)
            prev, cur = cur, nn
        if cur in junc:
            arc = sum(np.linalg.norm(corners[rc[c][1]] - corners[rc[c][0]]) for c in chain)
            cv = corners[cur] - corners[j]
            ch = np.linalg.norm(cv)
            if ch / max(arc, 1e-30) > 0.99 and abs(cv[1]) / max(ch, 1e-30) > 0.6:
                is_web[chain] = True
skin = np.where(~is_web)[0]
ang = np.arctan2(mid[skin, 1] - cen[1], mid[skin, 0] - cen[0])
order = skin[np.argsort(ang)]
L = np.linalg.norm(corners[rc[order, 1]] - corners[rc[order, 0]], axis=1)
s_arc = (np.cumsum(L) - 0.5 * L) * 1e3     # mm

A = np.zeros(len(order)); Bt = np.zeros(len(order)); Ct = np.zeros(len(order))
A5 = np.zeros(len(order)); B5 = np.zeros(len(order)); C5 = np.zeros(len(order))
for i, e in enumerate(order):
    Xe = nodes[quads[e]]; e3e = B['re3'][e]
    BDe, BDh, BDl, *_ = quad_ops_indep(Xe, e3e, 0.0, 0.0, float(B['k22'][e]),
                                       B['cross'], B['ax'])
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6, c1 * 6:c1 * 6 + 6, c0 * 6:c0 * 6 + 6]
    A[i] = (BDe @ st_m)[2]; Bt[i] = (BDh @ aA[g])[2]; Ct[i] = (BDl @ aB[g])[2]
    A5[i] = (BDe @ st_m)[5]; B5[i] = (BDh @ aA[g])[5]; C5[i] = (BDl @ aB[g])[5]

tot = A + Bt + Ct


def jumpstat(v, name):
    dj = np.abs(np.diff(v))
    print('%-18s mean|jump| %.4e   mean|value| %.4e   ratio %.2f'
          % (name, dj.mean(), np.abs(v).mean() + 1e-30, dj.mean() / (np.abs(v).mean() + 1e-30)))


print('=== membrane shear row 2eps12: element-midpoint values, loop order ===')
jumpstat(A, 'A macro (BDe)')
jumpstat(Bt, 'B contour-deriv')
jumpstat(Ct, 'C axial-deriv')
jumpstat(tot, 'total')
print('=== twist row 2k12 ===')
jumpstat(A5, 'A5'); jumpstat(B5, 'B5'); jumpstat(C5, 'C5')

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(s_arc, A, label='A macro (BDe)')
ax[0].plot(s_arc, Bt, label='B contour-derivative (BDh)')
ax[0].plot(s_arc, Ct, label='C axial-derivative (BDl)')
ax[0].set_ylabel('2eps12 contributions')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[1].plot(s_arc, tot, 'k', label='total 2eps12 (element mid)')
sm = tot.copy()
sm[1:-1] = 0.25 * tot[:-2] + 0.5 * tot[1:-1] + 0.25 * tot[2:]
ax[1].plot(s_arc, sm, 'r', label='1-2-1 smoothed')
ax[1].set_ylabel('2eps12'); ax[1].set_xlabel('arc (mm)')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'out', 'debug_s12_decomp.png'), dpi=150)
print('wrote out/debug_s12_decomp.png')
