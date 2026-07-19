"""Check whether the conformal-mesh OML coords (build_cross_section) and the 1-D-yaml ring coords
(B['corners'], the projection target) share a coordinate frame.  A mismatch would mis-map the dehom."""
import os, sys, numpy as np
BUCK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BUCK)
import blade_buckling as bb

for i in [10, 25, 40]:
    r = i / 50.0
    oml = bb.resample(np.asarray(bb.build_cross_section(bb.blade, r=r)["nodes"], float), bb.N)
    B = bb.homogenize_station(i)
    cor = np.asarray(B["corners"])
    print("station %d (r=%.2f):" % (i, r))
    print("  OML   bbox x[%.3f,%.3f] y[%.3f,%.3f]  centroid (%.3f,%.3f)  n=%d"
          % (oml[:, 0].min(), oml[:, 0].max(), oml[:, 1].min(), oml[:, 1].max(),
             oml[:, 0].mean(), oml[:, 1].mean(), len(oml)))
    print("  ring  bbox x[%.3f,%.3f] y[%.3f,%.3f]  centroid (%.3f,%.3f)  n=%d"
          % (cor[:, 0].min(), cor[:, 0].max(), cor[:, 1].min(), cor[:, 1].max(),
             cor[:, 0].mean(), cor[:, 1].mean(), len(cor)))
