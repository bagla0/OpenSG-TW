"""Cylinder axial buckling with the STANDARD (Abaqus/Ansys benchmark) supported-end BCs.

The classical  N_cr = E t^2 / (R sqrt(3(1-nu^2)))  is derived for SIMPLY-SUPPORTED ends of a
moderately-long cylinder (Batdorf Z large), minimized over the Koiter (m,n) circle.  A clamped-
FREE (cantilever) end has a free-edge boundary layer that buckles at a LOWER load -- physically
correct, but NOT what the classical formula gives.  So we validate the formulation against
classical using SS / clamped-clamped ends (as Abaqus/Ansys verification cases do), then report
the cantilever separately to show the (expected) free-edge reduction."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # examples/buckling
import shell_buckling as sb

E, nu, R, t, L = 200e9, 0.3, 1.0, 0.02, 2.0
Ncl = E * t ** 2 / (R * np.sqrt(3 * (1 - nu ** 2)))
Z = L ** 2 / (R * t) * np.sqrt(1 - nu ** 2)
print("classical N_cr = %.4e N/m   Batdorf Z=%.1f   (R/t=%.0f  L/R=%.1f)\n" % (Ncl, Z, R / t, L / R))


def build(nc, nl):
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    xs = np.linspace(0, L, nl + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])]
                      for i in range(nl + 1) for j in range(nc)])
    idx = lambda i, j: i * nc + (j % nc)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nl) for j in range(nc)])
    return nodes, quads, idx


def run(nc, nl, bc):
    nodes, quads, idx = build(nc, nl)
    ABD, Gs = sb._iso_ABD(E, nu, t)
    ne = len(quads)
    ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec_e = np.repeat(np.array([-1.0, 0.0, 0.0])[None], ne, 0)     # uniform axial compression
    fixed = []
    for j in range(nc):
        r0, rL = idx(0, j), idx(nl, j)
        if bc == "SS":                 # radial+circ (uy,uz) fixed both ends; axial (ux) anchored at root; rotations free
            fixed += [6 * r0 + 1, 6 * r0 + 2, 6 * r0 + 0, 6 * rL + 1, 6 * rL + 2]
        elif bc == "clamp-clamp":      # all 6 DOF at both end rings
            fixed += [6 * r0 + k for k in range(6)] + [6 * rL + k for k in range(6)]
        elif bc == "clamp-free":       # all 6 DOF at root ring only; tip free (the blade/cantilever case)
            fixed += [6 * r0 + k for k in range(6)]
    fixed = np.unique(fixed)
    loads, _ = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec_e, fixed, n_modes=6)
    print("  %-12s nc=%d nl=%d : FE=%.4e  ratio=%.3f   first4=%s"
          % (bc, nc, nl, loads[0], loads[0] / Ncl, np.array2string(loads[:4], precision=3)))
    return loads[0] / Ncl


nc, nl = 160, 80
print("mesh nc=%d nl=%d  (circ elem=%.3f  axial elem=%.3f  sqrt(Rt)=%.3f):"
      % (nc, nl, 2 * np.pi * R / nc, L / nl, np.sqrt(R * t)))
for bc in ["SS", "clamp-clamp", "clamp-free"]:
    run(nc, nl, bc)
