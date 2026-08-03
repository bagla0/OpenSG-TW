"""solid_vs_fsm.py -- APPLES-TO-APPLES: 3-D SOLID (FEniCSx) vs OpenSG RM-FSM, identical load, center ref.

WHY.  Previous comparisons were not fair: the solid solved for its OWN pre-buckling stress while the FSM was
fed an ASSUMED analytic membrane force N = -P/(perimeter*cos a).  Any difference then mixed
"wrong pre-stress" with "wrong buckling formulation", which is exactly the ambiguity that let earlier bugs
hide.  Here the solid's static solve is the single source of truth for the load:

    1. solid: apply a UNIT total axial force, solve the linear static problem -> sigma
    2. extract the wall membrane resultant  N11(x_i, s_j) = int sigma_xx dz  through the wall thickness
    3. solid buckling eigenvalue (no rotational DOF at all -> no drilling penalty, no fiber-rotation map)
    4. FSM driven by THAT SAME N11, on the SAME mid-surface contour (center ref) and the same wall ABD
    5. compare -- the only remaining difference is the buckling formulation itself

CENTER REF: the solid wall is built as mid-surface +- t/2, and the FSM ring IS that mid-surface contour, so
both reference the same surface.
"""
import os, sys, time
import numpy as np
import ufl
import basix
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import LinearProblem, assemble_matrix

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fsm_buckling as fsm

E, NU, T, L = 200e9, 0.3, 0.02, 2.0


def contour(kind, s, nper):
    """mid-surface points (y,z) + INWARD unit normals."""
    if kind == "circle":
        th = np.linspace(0, 2 * np.pi, nper, endpoint=False)
        return np.column_stack([s * np.cos(th), s * np.sin(th)]), -np.column_stack([np.cos(th), np.sin(th)])
    nps = nper // 4
    cor = [(-s / 2, -s / 2), (s / 2, -s / 2), (s / 2, s / 2), (-s / 2, s / 2)]
    P, N = [], []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        d = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-d[1], d[0]]); mid = 0.5 * (P0 + P1)
        if np.dot(-mid, nrm) < 0:
            nrm = -nrm
        for j in range(nps):
            P.append(P0 + (j / nps) * (P1 - P0)); N.append(nrm)
    return np.array(P), np.array(N)


def build(kind, s1, s2, nper, nspan, nthk):
    xs = np.linspace(0.0, L, nspan + 1); zk = np.linspace(-0.5, 0.5, nthk + 1)
    X = np.zeros(((nspan + 1) * nper * (nthk + 1), 3))
    nid = lambda i, j, k: (i * nper + (j % nper)) * (nthk + 1) + k
    for i, x in enumerate(xs):
        P, nin = contour(kind, s1 + (s2 - s1) * x / L, nper)
        for j in range(nper):
            for k, zz in enumerate(zk):
                p = P[j] + (zz * T) * (-nin[j])
                X[nid(i, j, k)] = (x, p[0], p[1])
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(nspan) for j in range(nper) for k in range(nthk)], dtype=np.int64)
    el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    return dmesh.create_mesh(MPI.COMM_WORLD, cells, X, ufl.Mesh(el))


def solid_run(dom, nper, nspan, nthk, n_modes=6, degree=2):
    """static under unit axial force -> N11 per (span,perimeter) column ; then buckling eigenvalues."""
    mu = E / (2 * (1 + NU)); lm = E * NU / ((1 + NU) * (1 - 2 * NU))
    eps = lambda v: ufl.sym(ufl.grad(v))
    sig = lambda v: lm * ufl.tr(eps(v)) * ufl.Identity(3) + 2 * mu * eps(v)
    V = fem.functionspace(dom, ("Lagrange", degree, (3,)))
    du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    xg = dom.geometry.x; x0, x1 = xg[:, 0].min(), xg[:, 0].max()
    fd = dom.topology.dim - 1
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
    a = ufl.inner(sig(du), eps(v)) * ufl.dx
    l = ufl.dot(fem.Constant(dom, np.array([-1.0 / A, 0.0, 0.0])), v) * ds(1)
    u0 = fem.Function(V)
    LinearProblem(a, l, bcs=bcs, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                      "pc_factor_mat_solver_type": "mumps"}).solve()

    # --- membrane N11 per (i,j) column: DG0 sigma_xx, mapped back through original_cell_index ---
    S = fem.functionspace(dom, ("DG", 0))
    sxx = fem.Function(S)
    sxx.interpolate(fem.Expression(sig(u0)[0, 0], S.element.interpolation_points()))
    orig = dom.topology.original_cell_index
    vals = sxx.x.array
    N11 = np.zeros((nspan, nper))
    dz = T / nthk
    for loc, og in enumerate(orig):
        i = og // (nper * nthk); rem = og % (nper * nthk); j = rem // nthk
        N11[i, j] += vals[loc] * dz

    # --- buckling: KG x = theta K x (B=K is SPD; GHEP needs the 2nd operator PD, KG is indefinite) ---
    kg = -ufl.inner(sig(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
    K = assemble_matrix(fem.form(a), bcs=bcs, diagonal=1.0); K.assemble()
    KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
    ep = SLEPc.EPS().create(dom.comm)
    ep.setOperators(KG, K); ep.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    ep.setType(SLEPc.EPS.Type.KRYLOVSCHUR); ep.setDimensions(n_modes, PETSc.DECIDE)
    ep.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE); ep.setTolerances(1e-8, 400)
    ep.solve()
    lam = []
    for i in range(ep.getConverged()):
        th = ep.getEigenvalue(i).real
        if np.isfinite(th) and abs(th) > 1e-14:
            lam.append(abs(1.0 / th))
    return np.sort(np.array(lam)), N11, A


def fsm_from_N(kind, s1, s2, N11, nper, nspan, M=16):
    """FSM per station, driven by the SOLID's OWN extracted N11 (same contour, same wall)."""
    ABD = fsm.iso_abd(E, NU, T)
    xs = 0.5 * (np.linspace(0, L, nspan + 1)[:-1] + np.linspace(0, L, nspan + 1)[1:])
    out = []
    for i in range(nspan):
        s = s1 + (s2 - s1) * xs[i] / L
        ring, strips = (fsm.cyl_ring(s, nper) if kind == "circle" else _sq_ring(s, nper))
        N_s = [np.array([N11[i, j], 0.0, 0.0]) for j in range(len(strips))]
        lam = np.asarray(fsm.solve_fsm_multi(ring, strips, [ABD] * len(strips), N_s, L, M, n_modes=2))
        if lam.size:
            out.append((xs[i], float(lam[0])))
    return out


def _sq_ring(a, n):
    nps = n // 4; cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    P = [np.array(cor[k], float) + (j / nps) * (np.array(cor[(k + 1) % 4], float) - np.array(cor[k], float))
         for k in range(4) for j in range(nps)]
    return np.array(P), np.array([[i, (i + 1) % (4 * nps)] for i in range(4 * nps)])


if __name__ == "__main__":
    NPER, NSPAN, NTHK = 64, 40, 2
    D = E * T**3 / (12 * (1 - NU**2))
    print("APPLES-TO-APPLES  solid(FEniCSx) vs RM-FSM, identical N, center ref   nper=%d nspan=%d nthk=%d"
          % (NPER, NSPAN, NTHK))
    for kind, s1, s2, label in [("square", 1.0, 1.0, "square prismatic"),
                                ("square", 1.0, 0.5, "square TAPERED"),
                                ("circle", 1.0, 1.0, "circle prismatic"),
                                ("circle", 1.0, 0.5, "circle TAPERED (cone)")]:
        t0 = time.time()
        dom = build(kind, s1, s2, NPER, NSPAN, NTHK)
        lam, N11, A = solid_run(dom, NPER, NSPAN, NTHK)
        if lam.size == 0:
            print("%-22s SOLID: no converged eigenvalues" % label); continue
        fs = fsm_from_N(kind, s1, s2, N11, NPER, NSPAN)
        xg, lg = min(fs, key=lambda r: r[1])
        # analytic where available (prismatic only)
        an = (16 * np.pi**2 * D / 1.0) if (kind == "square" and s2 == 1.0) else (
            (E * T**2 / (1.0 * np.sqrt(3 * (1 - NU**2)))) * 2 * np.pi if (kind == "circle" and s2 == 1.0) else None)
        print("\n%-22s  (%.0fs)  end area A=%.4e" % (label, time.time() - t0, A))
        print("   SOLID lam1 = %.5e %s" % (lam[0], ("| analytic %.5e -> %.4f" % (an, lam[0] / an)) if an else ""))
        print("   FSM   lam1 = %.5e  at x=%.2f      FSM/SOLID = %.4f" % (lg, xg, lg / lam[0]))
        print("   N11 range over the wall: %.4e .. %.4e  (mean %.4e)" % (N11.min(), N11.max(), N11.mean()))
