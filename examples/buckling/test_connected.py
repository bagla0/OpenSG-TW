"""test_connected.py -- CONNECTED 4-section FSM vs the 3-D solid, tapered square, isotropic.

1) SANITY: on a PRISMATIC member the connected model must reproduce the per-station (multi-harmonic)
   answer -- with no x-variation the harmonics decouple and the x-integral gives L/2 * delta_mm'.
2) TAPER: build the 4 cross-sections, give each the solid's OWN extracted N, run the connected model,
   and compare with (a) the per-station minimum over the same 4 sections and (b) the 3-D solid.
3) MODE: reconstruct the connected mode's amplitude vs span and compare with where the solid buckles
   (solid peaks at x~0.5-0.7, spread x~0.2-1.4).  A correct connected model must localize similarly.
"""
import os, sys, time
import numpy as np
import ufl, basix
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import LinearProblem, assemble_matrix

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fsm_buckling as fsm

E, NU, T, L = 200e9, 0.3, 0.02, 2.0
S1, S2 = 1.0, 0.5
NPER, NSPAN, NTHK = 64, 40, 2
M, NG = 8, 32
ABD = fsm.iso_abd(E, NU, T)
ax = lambda x: S1 + (S2 - S1) * x / L


def sq_ring(a, n):
    nps = n // 4
    cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    P, Nin = [], []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        d = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-d[1], d[0]]); mid = 0.5 * (P0 + P1)
        if np.dot(-mid, nrm) < 0:
            nrm = -nrm
        for j in range(nps):
            P.append(P0 + (j / nps) * (P1 - P0)); Nin.append(nrm)
    P = np.array(P)
    return P, np.array([[i, (i + 1) % len(P)] for i in range(len(P))]), np.array(Nin)


# ---------------------------------------------------------------- 1) prismatic sanity check
print("=" * 88)
print("1) SANITY  prismatic square: connected model must equal the per-station multi-harmonic answer")
ring0, st0, _ = sq_ring(1.0, NPER)
Nuni = [np.array([-1.0 / 4.0, 0.0, 0.0])] * len(st0)
lam_ps = np.asarray(fsm.solve_fsm_multi(ring0, st0, [ABD] * len(st0), Nuni, L, M, n_modes=2))
secs_pris = [(x, ring0, [ABD] * len(st0), Nuni) for x in (0.0, L / 3, 2 * L / 3, L)]
t0 = time.time()
lam_cn = np.asarray(fsm.solve_fsm_connected(secs_pris, L, M, n_modes=4, ngauss=NG))
print("   per-station multi-harmonic : %.6e" % lam_ps[0])
print("   CONNECTED (4 identical sec): %.6e   ratio=%.5f   (%.0fs)"
      % (lam_cn[0], lam_cn[0] / lam_ps[0], time.time() - t0))

# ---------------------------------------------------------------- solid: static, N, buckling, mode
def build():
    xs = np.linspace(0.0, L, NSPAN + 1); zk = np.linspace(-0.5, 0.5, NTHK + 1)
    X = np.zeros(((NSPAN + 1) * NPER * (NTHK + 1), 3))
    nid = lambda i, j, k: (i * NPER + (j % NPER)) * (NTHK + 1) + k
    for i, x in enumerate(xs):
        P, _, nin = sq_ring(ax(x), NPER)
        for j in range(NPER):
            for k, zz in enumerate(zk):
                p = P[j] + (zz * T) * (-nin[j]); X[nid(i, j, k)] = (x, p[0], p[1])
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(NSPAN) for j in range(NPER) for k in range(NTHK)], dtype=np.int64)
    return dmesh.create_mesh(MPI.COMM_WORLD, cells,
                             X, ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))))


print("\n2) SOLID reference (P2, unit total axial force)")
dom = build()
mu = E / (2 * (1 + NU)); lm = E * NU / ((1 + NU) * (1 - 2 * NU))
eps = lambda v: ufl.sym(ufl.grad(v))
sig = lambda v: lm * ufl.tr(eps(v)) * ufl.Identity(3) + 2 * mu * eps(v)
V = fem.functionspace(dom, ("Lagrange", 2, (3,)))
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
xg_ = dom.geometry.x; x0, x1 = xg_[:, 0].min(), xg_[:, 0].max(); fd = dom.topology.dim - 1
lf = dmesh.locate_entities_boundary(dom, fd, lambda p: np.isclose(p[0], x0, atol=1e-9))
rf = dmesh.locate_entities_boundary(dom, fd, lambda p: np.isclose(p[0], x1, atol=1e-9))
z0 = fem.Constant(dom, 0.0)
bcs = [fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(1), fd, lf), V.sub(1)),
       fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(2), fd, lf), V.sub(2)),
       fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(0), fd, lf), V.sub(0)),
       fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(1), fd, rf), V.sub(1)),
       fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(2), fd, rf), V.sub(2))]
mt = dmesh.meshtags(dom, fd, np.sort(rf), np.full(len(rf), 1, np.int32))
ds = ufl.Measure("ds", domain=dom, subdomain_data=mt)
A = dom.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(dom, 1.0) * ds(1))), op=MPI.SUM)
a_f = ufl.inner(sig(du), eps(v)) * ufl.dx
l_f = ufl.dot(fem.Constant(dom, np.array([-1.0 / A, 0.0, 0.0])), v) * ds(1)
u0 = fem.Function(V)
LinearProblem(a_f, l_f, bcs=bcs, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                      "pc_factor_mat_solver_type": "mumps"}).solve()
S = fem.functionspace(dom, ("DG", 0)); sxx = fem.Function(S)
sxx.interpolate(fem.Expression(sig(u0)[0, 0], S.element.interpolation_points()))
N11 = np.zeros((NSPAN, NPER)); dz = T / NTHK
for loc, og in enumerate(dom.topology.original_cell_index):
    N11[og // (NPER * NTHK), (og % (NPER * NTHK)) // NTHK] += sxx.x.array[loc] * dz
kg = -ufl.inner(sig(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
ep = SLEPc.EPS().create(dom.comm); ep.setOperators(KG, K)
ep.setProblemType(SLEPc.EPS.ProblemType.GHEP); ep.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
ep.setDimensions(4, PETSc.DECIDE); ep.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
ep.setTolerances(1e-8, 400); ep.solve()
ls, vs = [], []
for i in range(ep.getConverged()):
    th = ep.getEigenvalue(i).real
    if np.isfinite(th) and abs(th) > 1e-14:
        vr = fem.Function(V); ep.getEigenvector(i, vr.x.petsc_vec); vr.x.scatter_forward()
        ls.append(abs(1.0 / th)); vs.append(vr)
o = int(np.argmin(ls)); lam_solid = ls[o]; vsol = vs[o]
print("   SOLID lam1 = %.5e" % lam_solid)

# ---------------------------------------------------------------- 3) the 4 sections, connected
xs4 = np.linspace(0, L, 4)
xmid = 0.5 * (np.linspace(0, L, NSPAN + 1)[:-1] + np.linspace(0, L, NSPAN + 1)[1:])
secs = []
for x in xs4:
    i = int(np.clip(np.searchsorted(xmid, x), 0, NSPAN - 1))
    P, st, _ = sq_ring(ax(x), NPER)
    secs.append((float(x), P, [ABD] * len(st), [np.array([N11[i, j], 0.0, 0.0]) for j in range(len(st))]))

print("\n3) TAPERED square, 4 sections")
per = []
for (x, P, Al, Nl) in secs:
    st = np.array([[i, (i + 1) % len(P)] for i in range(len(P))])
    lam = np.asarray(fsm.solve_fsm_multi(P, st, Al, Nl, L, M, n_modes=2))
    per.append(float(lam[0])); print("   per-station x=%.3f a=%.4f : %.5e" % (x, ax(x), per[-1]))
t0 = time.time()
lam_c, Vc = fsm.solve_fsm_connected(secs, L, M, n_modes=4, ngauss=NG, return_vecs=True)
print("\n   per-station MIN over 4      : %.5e   /solid = %.4f" % (min(per), min(per) / lam_solid))
print("   CONNECTED  (4 sec coupled)  : %.5e   /solid = %.4f   (%.0fs)"
      % (lam_c[0], lam_c[0] / lam_solid, time.time() - t0))
print("   SOLID                       : %.5e" % lam_solid)

# ---------------------------------------------------------------- 4) mode localization comparison
print("\n4) MODE localization along the span   (connected FSM vs solid)")
xs_p = np.linspace(0, L, 41)
W = np.zeros_like(xs_p)
for mi in range(M):
    km = (mi + 1) * np.pi / L
    amp = np.abs(Vc[mi, :, 2, 0]).max()          # w-amplitude of harmonic mi, mode 0
    W += amp * np.abs(np.sin(km * xs_p))
W /= (W.max() + 1e-30)
vv = vsol.x.array.reshape(-1, 3); Vx = V.tabulate_dof_coordinates()
mag = np.linalg.norm(vv[:, 1:], axis=1); nb = 10; edges = np.linspace(0, L, nb + 1); amp_s = np.zeros(nb)
for b in range(nb):
    m_ = (Vx[:, 0] >= edges[b]) & (Vx[:, 0] < edges[b + 1] + 1e-12)
    amp_s[b] = mag[m_].max() if m_.any() else 0.0
amp_s /= (amp_s.max() + 1e-30)
print("      x-band      solid   connectedFSM")
for b in range(nb):
    xc = 0.5 * (edges[b] + edges[b + 1]); wc = np.interp(xc, xs_p, W)
    print("   [%.2f,%.2f]   %-6.3f  %-6.3f  %s" % (edges[b], edges[b + 1], amp_s[b], wc,
                                                   "#" * int(30 * wc)))
