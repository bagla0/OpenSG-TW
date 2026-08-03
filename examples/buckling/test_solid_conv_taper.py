"""test_solid_conv_taper.py -- is the TAPERED-square solid reference itself converged?

This decides whether the connected-FSM gap (~0.89) is real.  On the PRISMATIC square the solid was 4% BELOW
the analytic k=4 at 64x40 and rose to 0.981 at 96x60 -- it converges UPWARD.  If the tapered solid does the
same, the converged target is HIGHER and the true FSM gap is larger than the 11% measured against the coarse
solid.  Refine in perimeter, span and through-thickness independently so the responsible direction is clear.
"""
import os, sys, time
import numpy as np
import ufl, basix
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import LinearProblem, assemble_matrix

E, NU, T, L = 200e9, 0.3, 0.02, 2.0
S1, S2 = 1.0, 0.5
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
    return np.array(P), np.array(Nin)


def offsets(nin):
    """MITRED wall offset. Offsetting each node along its OWN face normal pinches the wall to zero
    effective thickness at every corner (the corner node's offset is tangential to the adjoining face),
    which hosts 4 spurious near-zero-energy modes and gets RELATIVELY worse under refinement -- that is
    what made this sweep diverge (-99.8% at 96x60, and worse under p-refinement, which no discretization
    error does). Offset along the angle bisector scaled by 1/(b.n) instead: the offset then lands on the
    intersection of the two offset lines, restoring exact perpendicular thickness on both faces."""
    n = len(nin); out = np.zeros_like(nin)
    for j in range(n):
        np_, nn = nin[(j - 1) % n], nin[j]
        b = np_ + nn; nb = np.linalg.norm(b)
        out[j] = nn if nb < 1e-12 else (b / nb) / max(np.dot(b / nb, nn), 1e-9)
    return out


def run(nper, nspan, nthk, deg=2):
    xs = np.linspace(0.0, L, nspan + 1); zk = np.linspace(-0.5, 0.5, nthk + 1)
    X = np.zeros(((nspan + 1) * nper * (nthk + 1), 3))
    nid = lambda i, j, k: (i * nper + (j % nper)) * (nthk + 1) + k
    for i, x in enumerate(xs):
        P, nin = sq_ring(ax(x), nper); off = offsets(nin)
        for j in range(nper):
            for k, zz in enumerate(zk):
                X[nid(i, j, k)] = (x, *(P[j] + (zz * T) * (-off[j])))
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(nspan) for j in range(nper) for k in range(nthk)], dtype=np.int64)
    dom = dmesh.create_mesh(MPI.COMM_WORLD, cells, X,
                            ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))))
    mu = E / (2 * (1 + NU)); lm = E * NU / ((1 + NU) * (1 - 2 * NU))
    ep_ = lambda v: ufl.sym(ufl.grad(v))
    sg = lambda v: lm * ufl.tr(ep_(v)) * ufl.Identity(3) + 2 * mu * ep_(v)
    V = fem.functionspace(dom, ("Lagrange", deg, (3,)))
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
    a_f = ufl.inner(sg(du), ep_(v)) * ufl.dx
    l_f = ufl.dot(fem.Constant(dom, np.array([-1.0 / A, 0.0, 0.0])), v) * ds(1)
    u0 = fem.Function(V)
    LinearProblem(a_f, l_f, bcs=bcs, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                          "pc_factor_mat_solver_type": "mumps"}).solve()
    kg = -ufl.inner(sg(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
    K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
    KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
    e2 = SLEPc.EPS().create(dom.comm); e2.setOperators(KG, K)
    e2.setProblemType(SLEPc.EPS.ProblemType.GHEP); e2.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    e2.setDimensions(4, PETSc.DECIDE); e2.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    e2.setTolerances(1e-8, 400); e2.solve()
    ls = []
    for i in range(e2.getConverged()):
        th = e2.getEigenvalue(i).real
        if np.isfinite(th) and abs(th) > 1e-14:
            ls.append(abs(1.0 / th))
    return (min(ls) if ls else np.nan), V.dofmap.index_map.size_global * 3


print("TAPERED square: is the SOLID reference converged?   (connected FSM ~ 2.708e7 at nsec=8)")
print("  nper nspan nthk deg |   ndof    | solid lam1      | vs 64/40/2")
base = None
for (npe, nsp, nth, dg) in [(64, 40, 2, 2), (96, 60, 2, 2), (128, 80, 2, 2), (96, 60, 3, 2), (96, 60, 2, 3)]:
    t0 = time.time()
    try:
        lam, nd = run(npe, nsp, nth, dg)
    except Exception as ex:
        print("  %4d %5d %4d %3d | FAILED %s" % (npe, nsp, nth, dg, type(ex).__name__)); continue
    if base is None:
        base = lam
    print("  %4d %5d %4d %3d | %9d | %.5e | %+.2f%%   (%.0fs)  FSM/solid=%.4f"
          % (npe, nsp, nth, dg, nd, lam, 100 * (lam / base - 1), time.time() - t0, 2.70809e7 / lam))
