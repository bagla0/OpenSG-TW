"""_extract_landmarks.py  --  ONE-OFF pre-extraction (NOT a committed tutorial).

The IEA-22 spanwise VABS dumps are enormous: iea_sNN.sg.SM (~42 MB) and .sg.U (~5 MB)
for each of 51 stations = ~875 MB total -- far too big for GitHub.  But the spanwise
tutorial (docs/tutorials/iea_spanwise.*) needs only, per station:

  * the near-root-band landmark point (y2,y3)        (max-y3 point with |y2|<band),
  * the VABS 3-D stress 6-vector at that landmark     (SM native column order),
  * the VABS warping displacement 3-vector there      (inverse-distance from .U),
  * the VABS Timoshenko 6x6 stiffness                 (.sg.K).

This script reproduces EXACTLY the landmark selection + benchmark reads that the
reference driver (dehom51/out/spanwise_dehom_oml/spanwise_center51.py) performs on the
big files, and packs them into ONE tiny file:

    dehom51/benchmark/spanwise_vabs_landmarks.npz   (< 1 MB)

Run it ONCE on the server where the big VABS dumps live:
    ~/miniconda3/envs/opensg_2_0/bin/python _extract_landmarks.py
"""
import os

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DEHOM = os.path.abspath(os.path.join(HERE, ".."))          # .../dehom51
VABS = os.path.join(DEHOM, "out", "VABS_iea51")


def block6(L, key):
    """Read the first 6x6 numeric block after the line containing ``key`` (VABS .sg.K)."""
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


idx, PT, VS, VU, KK = [], [], [], [], []
for i in range(51):
    smp = os.path.join(VABS, "iea_s%02d.sg.SM" % i)
    up = os.path.join(VABS, "iea_s%02d.sg.U" % i)
    kp = os.path.join(VABS, "iea_s%02d.sg.K" % i)
    if not all(os.path.exists(p) for p in (smp, up, kp)):
        print("s%02d SKIP (missing dump)" % i, flush=True)
        continue
    SM = np.loadtxt(smp, skiprows=2)
    Uu = np.loadtxt(up)
    Kv = block6([l for l in open(kp).read().splitlines()], "Timoshenko Stiffness Matrix")
    # --- landmark: max-y3 (suction-crown) point inside the near-root |y2|<band strip ---
    band = 0.10
    sel = np.abs(SM[:, 0]) < band
    if sel.sum() < 3:
        sel = np.abs(SM[:, 0]) < 0.30
    cand = np.where(sel)[0]
    itop = int(cand[np.argmax(SM[cand, 1])])
    pt = SM[itop, :2]
    Vs = SM[itop, 2:8]                                    # VABS stress 6-vec (SM native order)
    # --- VABS warping disp at the landmark: 4-NN inverse-distance from .sg.U ---
    uxy = Uu[:, 1:3]
    uv = Uu[:, 3:6]
    dU, iU = cKDTree(uxy).query(pt[None], k=4)
    wv = 1.0 / (dU + 1e-8 * (dU.sum(1, keepdims=True) + 1e-30))
    wv /= wv.sum(1, keepdims=True)
    Vu = np.einsum("pk,pkj->pj", wv, uv[iU])[0]
    idx.append(i)
    PT.append(pt)
    VS.append(Vs)
    VU.append(Vu)
    KK.append(Kv)
    print("s%02d ok  pt=(%.4f,%.4f)  s11=%.3e MPa  u3=%.3e  EA=%.3e"
          % (i, pt[0], pt[1], Vs[0] / 1e6, Vu[2], Kv[0, 0]), flush=True)

idx = np.array(idx, int)
out = os.path.join(HERE, "spanwise_vabs_landmarks.npz")
np.savez(out, idx=idx, pt=np.array(PT), VS=np.array(VS), VU=np.array(VU), K=np.array(KK))
sz = os.path.getsize(out) / 1e6
print("\nwrote %s  (%d stations, %.3f MB)" % (out, len(idx), sz))
