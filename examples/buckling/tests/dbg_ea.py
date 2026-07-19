"""Is the emitted mid-ref ABD (used in N=A@s6) consistent with the homogenized beam stiffness?
Compare int(A11) ds over the section to the homogenized EA=C6[0,0].  A mismatch = the N scale bug."""
import os, sys, numpy as np
BUCK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BUCK)
import blade_buckling as bb

for i in [10, 25, 40]:
    P, A, G = bb.station_abd(i)                                # per-elem ABD (emitted)
    B = bb.homogenize_station(i)
    C6 = np.asarray(B["Timo"])
    sec = bb.sec_elems
    L = np.linalg.norm(P[sec[:, 1]] - P[sec[:, 0]], axis=1)
    A11 = A[:, 0, 0]
    intA11 = float((A11 * L).sum())
    print("s%02d: EA=C6[0,0]=%.4e   int A11 ds=%.4e   ratio int/EA=%.3f   perimeter=%.3f m"
          % (i, C6[0, 0], intA11, intA11 / C6[0, 0], L.sum()))
