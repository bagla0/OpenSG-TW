'''loft_bin_demo.py -- first ~2 segments: build REAL cross-sections continuously along z (windIO is
continuous, so no perimeter interpolation), loft a QUAD OML shell, and bin the beam axis (z=0 at root)
into regular 2 m segments -> exact bin surface area -> flap force. This is Camarena's bin method with
the actual lofted surface, using only OpenSG's build_cross_section.'''
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser('~/OpenSG-TW-claude/third_party/OpenSG_io'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io.converter import load_blade, build_cross_section

L = 138.204
P_PA = 1500.0
N = 120                                   # points around each contour (for correspondence in the loft)
blade = load_blade(os.path.expanduser(
    '~/OpenSG-TW-claude/examples/data/iea_all_stations/IEA-22-280-RWT.yaml'))


def contour_at(z):
    """REAL section built at span z (any z; windIO continuous). Resample OML to N arc-even points."""
    xy = np.asarray(build_cross_section(blade, z / L)['xy'], float)
    cl = np.vstack([xy, xy[0]])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1])))]
    s = np.linspace(0.0, d[-1], N, endpoint=False)
    return np.column_stack([np.interp(s, d, cl[:, 0]), np.interp(s, d, cl[:, 1])]), float(d[-1])


zlev = np.arange(0.0, 8.001, 1.0)          # z-levels every 1 m across the first ~2 segments
C, P = [], []
for z in zlev:
    c, p = contour_at(z); C.append(c); P.append(p)
C = np.array(C); nz = len(zlev)

nodes = np.array([[C[i, k, 0], C[i, k, 1], zlev[i]] for i in range(nz) for k in range(N)])
quads = np.array([[i * N + k, i * N + (k + 1) % N, (i + 1) * N + (k + 1) % N, (i + 1) * N + k]
                  for i in range(nz - 1) for k in range(N)])
p0, p1, p2, p3 = (nodes[quads[:, j]] for j in range(4))
qA = 0.5 * np.linalg.norm(np.cross(p1 - p0, p3 - p0), axis=1) + \
     0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1)
qZ = nodes[quads, 2].mean(1)

print("built-section perimeter P(z) [m] (exact, no interpolation):")
print("  " + "  ".join("z%.0f=%.2f" % (z, p) for z, p in zip(zlev, P)))
print("\nlofted QUAD OML shell (first ~8 m): %d nodes, %d quads" % (len(nodes), len(quads)))
print("\nBIN = regular 2 m segment on the beam axis (z, root=0) -> surface area -> Fx at %.0f Pa:" % P_PA)
print("  %-9s %12s %10s %14s" % ("bin z[m]", "quad-area", "int P dz", "Fx=p*area[N]"))
for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8)]:
    A_q = qA[(qZ >= lo) & (qZ < hi)].sum()                          # sum of lofted quad areas in the bin
    A_p = np.trapz([contour_at(z)[1] for z in np.linspace(lo, hi, 9)], np.linspace(lo, hi, 9))  # int P dz
    print("  %2.0f - %2.0f  %12.3f %10.3f %14.1f" % (lo, hi, A_q, A_p, P_PA * A_q))
