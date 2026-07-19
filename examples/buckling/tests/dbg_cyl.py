"""Diagnose the cylinder axial-buckling under-prediction (FE ~0.38x classical).
Separates mesh-resolution from BC/formulation cause."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # examples/buckling
import shell_buckling as sb

E, nu, R, t = 200e9, 0.3, 1.0, 0.02
Ncl = E * t ** 2 / (R * np.sqrt(3 * (1 - nu ** 2)))
print("classical N_cr = %.4e  (sqrt(Rt)=%.4f)\n" % (Ncl, np.sqrt(R * t)))


def build(nc, nl, L):
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    xs = np.linspace(0, L, nl + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])]
                      for i in range(nl + 1) for j in range(nc)])
    idx = lambda i, j: i * nc + (j % nc)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nl) for j in range(nc)])
    return nodes, quads, idx


def run(nc, nl, L, bc="clamp-free", kdr_scale=1e-3, tag=""):
    nodes, quads, idx = build(nc, nl, L)
    ABD, Gs = sb._iso_ABD(E, nu, t)
    ne = len(quads)
    ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec_e = np.repeat(np.array([-1.0, 0.0, 0.0])[None], ne, 0)
    fixed = [6 * idx(0, j) + k for j in range(nc) for k in range(6)]
    if bc == "clamp-clamp":
        fixed += [6 * idx(nl, j) + k for j in range(nc) for k in range(6)]
    elif bc == "clamp-ring":            # tip: block radial+circ (v,w in-plane of ring) but free axial+rot
        fixed += [6 * idx(nl, j) + k for j in range(nc) for k in (1, 2)]
    fixed = np.unique(fixed)
    # temporarily override the drilling scale inside solve_buckling via monkeypatch of module const
    sb._KDR_SCALE = kdr_scale
    loads, _ = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec_e, fixed, n_modes=6)
    print("  %-22s nc=%3d nl=%3d L=%.1f kdr=%.0e : FE=%.4e  ratio=%.3f  modes=%s"
          % (tag, nc, nl, L, kdr_scale, loads[0], loads[0] / Ncl,
             np.array2string(loads[:4], precision=3)))
    return loads[0] / Ncl


print("[1] mesh convergence (clamp-free, L=2):")
for nc, nl in [(80, 40), (120, 60), (160, 80), (240, 120)]:
    run(nc, nl, 2.0, tag="mesh")

print("\n[2] boundary condition (nc=160 nl=80, L=2):")
for bc in ["clamp-free", "clamp-ring", "clamp-clamp"]:
    run(160, 80, 2.0, bc=bc, tag=bc)

print("\n[3] drilling penalty (clamp-clamp, nc=160 nl=80):")
for k in [1e-4, 1e-3, 1e-2, 1e-1]:
    run(160, 80, 2.0, bc="clamp-clamp", kdr_scale=k, tag="kdr")

print("\n[4] longer cylinder clamp-clamp (nc=200):")
for nl, L in [(120, 4.0), (160, 6.0)]:
    run(200, nl, L, bc="clamp-clamp", tag="long")
