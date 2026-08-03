"""test_connected_conv.py -- close the remaining gap on the CONNECTED tapered-square model.

Established: connected == per-station EXACTLY on a prismatic member (ratio 1.00000), and on the taper it
lifts FSM/solid from 0.770 (per-station min) to 0.901.  Remaining ~10% candidates:
   (a) harmonic truncation M      -- the buckle must be built from a finite sin/cos series
   (b) number of sections nsec    -- geometry/N are linearly interpolated between supplied sections
   (c) Kirchhoff strip (no transverse shear) -- a floor the FSM cannot cross
   (d) the SOLID's own discretization (~4% under-converged at this mesh)
Sweeping (a) and (b) separates them: if it converges in M and nsec and stops short of 1.0, the residual is
(c)+(d).

Also FIXES the mode reconstruction.  The previous plot summed |amp_m|*|sin(k_m x)|, which is symmetric BY
CONSTRUCTION and destroyed the phase -- it wrongly showed a symmetric mode.  Correct: sum the SIGNED
contributions w(x,xi) = sum_m W_m(xi) sin(k_m x) first, then take the magnitude over the contour.
The solid N11 and eigenvalue are cached so the 250 s solid solve runs only once.
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
CACHE = os.path.join(HERE, "data", "sqtaper_solid.npz")
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


def solid():
    if os.path.exists(CACHE):
        d = np.load(CACHE); print("   (solid loaded from cache)"); return d["N11"], float(d["lam"]), d["amp"]
    xs = np.linspace(0.0, L, NSPAN + 1); zk = np.linspace(-0.5, 0.5, NTHK + 1)
    X = np.zeros(((NSPAN + 1) * NPER * (NTHK + 1), 3))
    nid = lambda i, j, k: (i * NPER + (j % NPER)) * (NTHK + 1) + k
    for i, x in enumerate(xs):
        P, _, nin = sq_ring(ax(x), NPER)
        for j in range(NPER):
            for k, zz in enumerate(zk):
                X[nid(i, j, k)] = (x, *(P[j] + (zz * T) * (-nin[j])))
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(NSPAN) for j in range(NPER) for k in range(NTHK)], dtype=np.int64)
    dom = dmesh.create_mesh(MPI.COMM_WORLD, cells, X,
                            ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))))
    mu = E / (2 * (1 + NU)); lm = E * NU / ((1 + NU) * (1 - 2 * NU))
    ep_ = lambda v: ufl.sym(ufl.grad(v))
    sg = lambda v: lm * ufl.tr(ep_(v)) * ufl.Identity(3) + 2 * mu * ep_(v)
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
    a_f = ufl.inner(sg(du), ep_(v)) * ufl.dx
    l_f = ufl.dot(fem.Constant(dom, np.array([-1.0 / A, 0.0, 0.0])), v) * ds(1)
    u0 = fem.Function(V)
    LinearProblem(a_f, l_f, bcs=bcs, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                          "pc_factor_mat_solver_type": "mumps"}).solve()
    S = fem.functionspace(dom, ("DG", 0)); sxx = fem.Function(S)
    sxx.interpolate(fem.Expression(sg(u0)[0, 0], S.element.interpolation_points()))
    N11 = np.zeros((NSPAN, NPER)); dz = T / NTHK
    for loc, og in enumerate(dom.topology.original_cell_index):
        N11[og // (NPER * NTHK), (og % (NPER * NTHK)) // NTHK] += sxx.x.array[loc] * dz
    kg = -ufl.inner(sg(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
    K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
    KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
    e2 = SLEPc.EPS().create(dom.comm); e2.setOperators(KG, K)
    e2.setProblemType(SLEPc.EPS.ProblemType.GHEP); e2.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    e2.setDimensions(4, PETSc.DECIDE); e2.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    e2.setTolerances(1e-8, 400); e2.solve()
    ls, vs = [], []
    for i in range(e2.getConverged()):
        th = e2.getEigenvalue(i).real
        if np.isfinite(th) and abs(th) > 1e-14:
            vr = fem.Function(V); e2.getEigenvector(i, vr.x.petsc_vec); vr.x.scatter_forward()
            ls.append(abs(1.0 / th)); vs.append(vr)
    o = int(np.argmin(ls)); lam = ls[o]
    vv = vs[o].x.array.reshape(-1, 3); Vx = V.tabulate_dof_coordinates()
    mag = np.linalg.norm(vv[:, 1:], axis=1); nb = 10; edges = np.linspace(0, L, nb + 1); amp = np.zeros(nb)
    for b in range(nb):
        m_ = (Vx[:, 0] >= edges[b]) & (Vx[:, 0] < edges[b + 1] + 1e-12)
        amp[b] = mag[m_].max() if m_.any() else 0.0
    amp /= (amp.max() + 1e-30)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, N11=N11, lam=lam, amp=amp)
    return N11, lam, amp


print("tapered square, connected-FSM convergence   (solid reference)")
N11, lam_solid, amp_s = solid()
xmid = 0.5 * (np.linspace(0, L, NSPAN + 1)[:-1] + np.linspace(0, L, NSPAN + 1)[1:])
print("   SOLID lam1 = %.5e\n" % lam_solid)


def mk_secs(nsec):
    out = []
    for x in np.linspace(0, L, nsec):
        i = int(np.clip(np.searchsorted(xmid, x), 0, NSPAN - 1))
        P, st, _ = sq_ring(ax(x), NPER)
        out.append((float(x), P, [ABD] * len(st), [np.array([N11[i, j], 0.0, 0.0]) for j in range(len(st))]))
    return out


print("   nsec   M   connected lam1    /solid    time")
best = None
for nsec in (4, 6, 8):
    for M in (8, 12, 16):
        t0 = time.time()
        r = fsm.solve_fsm_connected(mk_secs(nsec), L, M, n_modes=3, ngauss=max(32, 4 * M), return_vecs=True)
        lam, Vc = r
        print("   %4d  %2d   %.5e     %.4f    %.0fs" % (nsec, M, lam[0], lam[0] / lam_solid, time.time() - t0))
        if nsec == 8 and M == 16:
            best = (lam, Vc)

# ---- corrected mode reconstruction: sum SIGNED harmonics first, then take magnitude ----
if best is not None:
    lam, Vc = best
    Mh = Vc.shape[0]
    xs_p = np.linspace(0, L, 101)
    Wsig = np.zeros((len(xs_p), Vc.shape[1]))
    for mi in range(Mh):
        km = (mi + 1) * np.pi / L
        Wsig += np.outer(np.sin(km * xs_p), Vc[mi, :, 2, 0])     # SIGNED sum, phase preserved
    prof = np.abs(Wsig).max(axis=1); prof /= (prof.max() + 1e-30)
    print("\n   MODE localization (corrected: signed harmonic sum, then magnitude)")
    print("      x-band       solid   connFSM")
    edges = np.linspace(0, L, 11)
    for b in range(10):
        xc = 0.5 * (edges[b] + edges[b + 1]); pc = np.interp(xc, xs_p, prof)
        print("   [%.2f,%.2f]    %-6.3f  %-6.3f %s" % (edges[b], edges[b + 1], amp_s[b], pc, "#" * int(30 * pc)))
