"""test_stacks.py -- is the anisotropic failure driven by A16 (membrane shear-extension) or D16 (bend-twist)?

The single-ply sweep showed orthotropic layups are accurate and every off-axis ply over-predicts, but a single
off-axis ply has A16 AND D16 large together, so it cannot separate them.  Laminate STACKS can:

  stack           A16      D16     what it isolates
  [0]             0        0       orthotropic control
  [0/90]          0        0       orthotropic, but with B coupling
  [-45]           max      max     both (the failing case, 1.3715)
  [+45/-45]       0        != 0    A16 KILLED, D16 retained  <-- THE DISCRIMINATOR
  [45/-45/-45/45] 0        != 0    balanced AND symmetric (B=0), D16 reduced further

Prediction if the mechanism is the skewed mode driven by MEMBRANE shear-extension coupling:
  [+45/-45] agrees (~1.0) despite being made of off-axis plies.
Prediction if BEND-TWIST coupling drives it:
  [+45/-45] still fails.

This also resolves a tension in the paper: a [+-45]s cylinder agrees at 1.014 against a 3-D SHELL reference
while the -45 single ply is 1.3715 against a 3-D SOLID. Those differ in laminate AND in reference model.
Running the balanced stacks against the SAME solid reference removes the second variable.
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

T, L, R = 0.02, 2.0, 1.0
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
NPER, NSPAN = 64, 40

STACKS = [
    ("[0]",          [0.0, 0.0]),
    ("[0/90]",       [0.0, 90.0]),
    ("[-45]",        [-45.0, -45.0]),
    ("[+45/-45]",    [45.0, -45.0]),
    ("[45/-45]s",    [45.0, -45.0, -45.0, 45.0]),
]


def ply_C(mat, ang):
    E1, E2, G12 = mat["E1"], mat["E2"], mat["G12"]; nu12 = mat["nu12"]
    E3 = E2; G13 = G12; G23 = 0.4 * E2; nu13 = nu12; nu23 = 0.4
    nu21 = nu12 * E2 / E1; nu31 = nu13 * E3 / E1; nu32 = nu23 * E3 / E2
    S = np.array([[1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                  [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                  [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                  [0, 0, 0, 1 / G23, 0, 0], [0, 0, 0, 0, 1 / G13, 0], [0, 0, 0, 0, 0, 1 / G12]])
    C = np.linalg.inv(S)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    Tm = np.array([[c*c, s*s, 0, 0, 0, 2*c*s], [s*s, c*c, 0, 0, 0, -2*c*s], [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, c, -s, 0], [0, 0, 0, s, c, 0], [-c*s, c*s, 0, 0, 0, c*c - s*s]])
    Rm = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tm) @ C @ Rm @ Tm @ np.linalg.inv(Rm)


def rot_global(Cl, e1, e2, e3):
    A = np.array([e1, e2, e3]); K1 = A**2
    K2 = np.array([[A[i, 1]*A[i, 2], A[i, 2]*A[i, 0], A[i, 0]*A[i, 1]] for i in range(3)])
    K3 = np.array([[A[1, j]*A[2, j], A[2, j]*A[0, j], A[0, j]*A[1, j]] for j in range(3)]).T
    K4 = np.array([[A[1, (j+1) % 3]*A[2, (j+2) % 3] + A[1, (j+2) % 3]*A[2, (j+1) % 3],
                    A[2, (j+1) % 3]*A[0, (j+2) % 3] + A[2, (j+2) % 3]*A[0, (j+1) % 3],
                    A[0, (j+1) % 3]*A[1, (j+2) % 3] + A[0, (j+2) % 3]*A[1, (j+1) % 3]] for j in range(3)]).T
    Tt = np.block([[K1, 2 * K2], [K3, K4]]); Rm = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tt) @ Cl @ Rm @ Tt @ np.linalg.inv(Rm)


def solid(angles):
    nthk = len(angles)                                    # one hex layer per ply
    xs = np.linspace(0.0, L, NSPAN + 1); zk = np.linspace(-0.5, 0.5, nthk + 1)
    th = np.linspace(0.0, 2 * np.pi, NPER, endpoint=False); ct, st_ = np.cos(th), np.sin(th)
    X = np.zeros(((NSPAN + 1) * NPER * (nthk + 1), 3))
    nid = lambda i, j, k: (i * NPER + (j % NPER)) * (nthk + 1) + k
    for i, x in enumerate(xs):
        for j in range(NPER):
            for k, zz in enumerate(zk):
                r = R + zz * T
                X[nid(i, j, k)] = (x, r * ct[j], r * st_[j])
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(NSPAN) for j in range(NPER) for k in range(nthk)], dtype=np.int64)
    dom = dmesh.create_mesh(MPI.COMM_WORLD, cells, X,
                            ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))))
    W = fem.functionspace(dom, basix.ufl.element("DG", dom.basix_cell(), 0, shape=(6, 6)))
    Cf = fem.Function(W); Cl = [ply_C(MAT, a) for a in angles]; arr = Cf.x.array.reshape(-1, 36)
    for loc, og in enumerate(dom.topology.original_cell_index):
        j = (og % (NPER * nthk)) // nthk
        k = og % nthk                                     # through-thickness layer -> ply, OML first
        arr[loc] = rot_global(Cl[k], np.array([1.0, 0, 0]), np.array([0.0, -st_[j], ct[j]]),
                              np.array([0.0, ct[j], st_[j]])).ravel()
    Cf.x.scatter_forward()
    vo = lambda e: ufl.as_vector([e[0, 0], e[1, 1], e[2, 2], 2*e[1, 2], 2*e[0, 2], 2*e[0, 1]])
    uv = lambda s: ufl.as_matrix([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
    ep_ = lambda v: ufl.sym(ufl.grad(v)); sg = lambda v: uv(ufl.dot(Cf, vo(ep_(v))))
    V = fem.functionspace(dom, ("Lagrange", 2, (3,)))
    du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    xg = dom.geometry.x; x0, x1 = xg[:, 0].min(), xg[:, 0].max(); fd = dom.topology.dim - 1
    lf = dmesh.locate_entities_boundary(dom, fd, lambda p: np.isclose(p[0], x0, atol=1e-9))
    rf = dmesh.locate_entities_boundary(dom, fd, lambda p: np.isclose(p[0], x1, atol=1e-9))
    z0 = fem.Constant(dom, 0.0)
    bcs = [fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(i), fd, lf), V.sub(i)) for i in range(3)]
    bcs += [fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(i), fd, rf), V.sub(i)) for i in (1, 2)]
    mt = dmesh.meshtags(dom, fd, np.sort(rf), np.full(len(rf), 1, np.int32))
    ds = ufl.Measure("ds", domain=dom, subdomain_data=mt)
    A_ = dom.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(dom, 1.0) * ds(1))), op=MPI.SUM)
    a_f = ufl.inner(sg(du), ep_(v)) * ufl.dx
    l_f = ufl.dot(fem.Constant(dom, np.array([-1.0 / A_, 0.0, 0.0])), v) * ds(1)
    u0 = fem.Function(V)
    LinearProblem(a_f, l_f, bcs=bcs, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                          "pc_factor_mat_solver_type": "mumps"}).solve()
    kg = -ufl.inner(sg(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
    K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
    KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
    ep = SLEPc.EPS().create(dom.comm); ep.setOperators(KG, K)
    ep.setProblemType(SLEPc.EPS.ProblemType.GHEP); ep.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    ep.setDimensions(8, PETSc.DECIDE); ep.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    ep.setTolerances(1e-9, 2000); ep.solve()
    ls = []
    for i in range(ep.getConverged()):
        t = ep.getEigenvalue(i).real
        if np.isfinite(t) and abs(t) > 1e-14:
            try:
                r_ = ep.computeError(i)
            except Exception:
                r_ = 0.0
            if r_ < 1e-4:
                ls.append(abs(1.0 / t))
    return min(ls) if ls else np.nan


strips = np.array([[i, (i + 1) % NPER] for i in range(NPER)])
thc = np.linspace(0.0, 2 * np.pi, NPER, endpoint=False)
P = np.c_[R * np.cos(thc), R * np.sin(thc)]
Nvec = np.tile(np.array([-1.0 / (2 * np.pi * R), 0.0, 0.0]), (NPER, 1))

print("Is the anisotropic failure driven by A16 (membrane) or D16 (bend-twist)?\n")
print("   stack          A16/A11   D16/D11   B_max/A11 |  FSM [N]        solid [N]      FSM/solid  time")
for name, angles in STACKS:
    ABD = np.asarray(fsm.clt_abd([(a, T / len(angles)) for a in angles], MAT), float)
    a16 = ABD[0, 2] / ABD[0, 0]; d16 = ABD[3, 5] / ABD[3, 3]
    bmax = np.abs(ABD[:3, 3:]).max() / ABD[0, 0]
    lam = np.asarray(fsm.solve_fsm_multi(P, strips, [ABD] * NPER, list(Nvec), L, 16, n_modes=3))
    lam = lam[np.isfinite(lam)]
    f = float(lam[0]) if lam.size else np.nan
    t0 = time.time()
    s = solid(angles)
    print("   %-12s  %8.4f  %8.4f  %8.4f  |  %.5e   %.5e   %8.4f  %4.0fs"
          % (name, a16, d16, bmax, f, s, f / s if s == s else np.nan, time.time() - t0))
print("""
   READ:  if [+45/-45] and [45/-45]s come back near 1.0 while [-45] stays at ~1.37, the driver is the
   MEMBRANE coupling A16 -- balancing the stack kills A16 and restores accuracy even though the plies are
   still off-axis, and even though D16 remains nonzero.  If they also fail, bend-twist D16 is implicated
   and balancing will not save a real blade layup.
""")
