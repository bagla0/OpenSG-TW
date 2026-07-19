"""Isolate the dehom moment reconstruction: one station, apply a pure unit flap moment, recover the wall
membrane N at each section element, and check whether int(N_span * z) ds reproduces the applied moment.
Also reports int(N) ds (should be ~0 for a pure moment) to expose any z-reference issue."""
import os, sys, numpy as np
BUCK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BUCK)
import blade_buckling as bb
import dehom_rm

i = 25
P, A, G = bb.station_abd(i)                                    # section coords + per-elem ABD
B = bb.homogenize_station(i)
sec = bb.sec_elems; NSE = len(sec)
mids = 0.5 * (P[sec[:, 0]] + P[sec[:, 1]])
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
mids = mids + (corners.mean(0) - P.mean(0))                   # align conformal OML frame -> 1-D ring frame

for label, FF in [("F1 axial", [1., 0, 0, 0, 0, 0]), ("M2 flap-bend", [0, 0, 0, 0, 1., 0]),
                  ("M3 edge-bend", [0, 0, 0, 0, 0, 1.])]:
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
    Fx = 0.0; My = 0.0; Mz = 0.0
    for se in range(NSE):
        e_ring, xi, pr = dehom_rm._project_point(corners, rc, mids[se])
        s6, _ = dehom_rm._rm_shell_strain(B, e_ring, xi, st_m, aA, aB)
        Nsp = (A[se][:3, :3] @ s6[:3] + A[se][:3, 3:] @ s6[3:6])[0]      # span membrane N (N11)
        L = np.linalg.norm(P[sec[se, 1]] - P[sec[se, 0]])
        y, z = mids[se]
        Fx += Nsp * L; My += Nsp * L * z; Mz += Nsp * L * y
    print("%-14s applied=%s  ->  int N ds=%.3e  int N z ds=%.3e  int N y ds=%.3e"
          % (label, FF, Fx, My, Mz))
