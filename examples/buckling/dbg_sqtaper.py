"""dbg_sqtaper.py -- DEEP end-to-end debug of the TAPERED SQUARE only, isotropic.

Questions being answered, in order:
  Q1  Are the FSM cross-sections actually DIFFERENT along the taper (a = 1.0 -> 0.5), or accidentally the
      same section repeated?
  Q2  Is the section the MID-REF (center) contour, matching the solid wall's mid-surface?
  Q3  What pre-buckling N does each section actually see -- solid-extracted vs the analytic -1/(4a)?
  Q4  What is each individual cross-section's FSM buckling load?
  Q5  How does the answer depend on the NUMBER of cross-sections (4, as intended, vs 10, 40)?
      More stations can only lower a min, so this is a methodology choice with a real effect.
  Q6  Where does the SOLID actually buckle along the span (mode localization)?
  Q7  Final apples-to-apples ratio with the intended 4-section method.
"""
import os, sys
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
D = E * T**3 / (12 * (1 - NU**2))
ABD = fsm.iso_abd(E, NU, T)


def sq_ring(a, n):
    """MID-REF square contour of side a (center ref: this is the wall mid-surface)."""
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


ax = lambda x: S1 + (S2 - S1) * x / L

print("=" * 92)
print("Q1/Q2  cross-sections along the taper (MID-REF square, side a(x)); are they DIFFERENT?")
for nsec in (4,):
    xs4 = np.linspace(0, L, nsec)
    for x in xs4:
        a = ax(x); P, st, _ = sq_ring(a, NPER)
        print("   x=%.3f  a=%.4f  perimeter=%.4f  contour extent y[%.3f,%.3f] z[%.3f,%.3f]  nnode=%d"
              % (x, a, 4 * a, P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), len(P)))

# ---------------------------------------------------------------- solid: mesh, static, N extraction, buckle
def build():
    xs = np.linspace(0.0, L, NSPAN + 1); zk = np.linspace(-0.5, 0.5, NTHK + 1)
    X = np.zeros(((NSPAN + 1) * NPER * (NTHK + 1), 3))
    nid = lambda i, j, k: (i * NPER + (j % NPER)) * (NTHK + 1) + k
    for i, x in enumerate(xs):
        P, _, nin = sq_ring(ax(x), NPER)
        for j in range(NPER):
            for k, zz in enumerate(zk):
                p = P[j] + (zz * T) * (-nin[j])
                X[nid(i, j, k)] = (x, p[0], p[1])
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(NSPAN) for j in range(NPER) for k in range(NTHK)], dtype=np.int64)
    el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    return dmesh.create_mesh(MPI.COMM_WORLD, cells, X, ufl.Mesh(el))


dom = build()
mu = E / (2 * (1 + NU)); lm = E * NU / ((1 + NU) * (1 - 2 * NU))
eps = lambda v: ufl.sym(ufl.grad(v))
sig = lambda v: lm * ufl.tr(eps(v)) * ufl.Identity(3) + 2 * mu * eps(v)
V = fem.functionspace(dom, ("Lagrange", 2, (3,)))
du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
xg = dom.geometry.x; x0, x1 = xg[:, 0].min(), xg[:, 0].max(); fd = dom.topology.dim - 1
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
orig = dom.topology.original_cell_index; vals = sxx.x.array
N11 = np.zeros((NSPAN, NPER)); dz = T / NTHK
for loc, og in enumerate(orig):
    i = og // (NPER * NTHK); j = (og % (NPER * NTHK)) // NTHK
    N11[i, j] += vals[loc] * dz
xmid = 0.5 * (np.linspace(0, L, NSPAN + 1)[:-1] + np.linspace(0, L, NSPAN + 1)[1:])

print("\nQ3  pre-buckling N11 per station: SOLID-extracted vs analytic -1/(4a)   (unit total axial force)")
print("      x      a      N_solid(mean)   N_solid(min..max)        N_analytic   solid/analytic")
for x in np.linspace(0, L, 4):
    i = int(np.clip(np.searchsorted(xmid, x), 0, NSPAN - 1))
    a = ax(xmid[i]); nan_ = -1.0 / (4 * a)
    print("   %6.3f %6.4f  %+.5e   %+.4e..%+.4e  %+.5e   %.4f"
          % (xmid[i], a, N11[i].mean(), N11[i].min(), N11[i].max(), nan_, N11[i].mean() / nan_))

kg = -ufl.inner(sig(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
ep = SLEPc.EPS().create(dom.comm); ep.setOperators(KG, K)
ep.setProblemType(SLEPc.EPS.ProblemType.GHEP); ep.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
ep.setDimensions(6, PETSc.DECIDE); ep.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
ep.setTolerances(1e-8, 400); ep.solve()
lam_s, vecs = [], []
for i in range(ep.getConverged()):
    th = ep.getEigenvalue(i).real
    if np.isfinite(th) and abs(th) > 1e-14:
        vr = fem.Function(V); ep.getEigenvector(i, vr.x.petsc_vec); vr.x.scatter_forward()
        lam_s.append(abs(1.0 / th)); vecs.append(vr)
o = np.argsort(lam_s); lam_s = np.array(lam_s)[o]; vecs = [vecs[k] for k in o]
print("\n   SOLID buckling lam1..3 = %s" % np.array2string(lam_s[:3], precision=5))

print("\nQ6  where does the SOLID buckle? (radial-ish amplitude of mode 1 vs span)")
vv = vecs[0].x.array.reshape(-1, 3)
Vx = V.tabulate_dof_coordinates()
nb = 10; edges = np.linspace(0, L, nb + 1); amp = np.zeros(nb)
mag = np.linalg.norm(vv[:, 1:], axis=1)
for b in range(nb):
    m = (Vx[:, 0] >= edges[b]) & (Vx[:, 0] < edges[b + 1] + 1e-12)
    amp[b] = mag[m].max() if m.any() else 0.0
amp /= (amp.max() + 1e-30)
for b in range(nb):
    print("   x=[%.2f,%.2f]  a=%.3f  |mode| %-5.3f %s" %
          (edges[b], edges[b + 1], ax(0.5 * (edges[b] + edges[b + 1])), amp[b], "#" * int(40 * amp[b])))

print("\nQ4/Q5  FSM per cross-section, and the effect of HOW MANY sections are used")
for nsec in (4, 10, 40):
    xs_s = np.linspace(0, L, nsec)
    rows = []
    for x in xs_s:
        i = int(np.clip(np.searchsorted(xmid, x), 0, NSPAN - 1))
        a = ax(xmid[i]); ring, st, _ = sq_ring(a, NPER)
        N_s = [np.array([N11[i, j], 0.0, 0.0]) for j in range(len(st))]
        lam = np.asarray(fsm.solve_fsm_multi(ring, st, [ABD] * len(st), N_s, L, 16, n_modes=2))
        rows.append((xmid[i], a, float(lam[0]) if lam.size else np.inf))
    best = min(rows, key=lambda r: r[2])
    if nsec == 4:
        for (xx, aa, ll) in rows:
            print("      x=%.3f a=%.4f  FSM lam1=%.5e" % (xx, aa, ll))
    print("   nsec=%2d -> min FSM = %.5e at x=%.3f (a=%.4f)   FSM/SOLID = %.4f"
          % (nsec, best[2], best[0], best[1], best[2] / lam_s[0]))
