"""dbg_corner.py -- are the 4 spurious degenerate modes living in the 4 CORNERS?

Evidence so far: material SPD (dbg_plyC), frame reflection is a no-op (dbg_frame), yet lam collapses ~100x
-- and it collapses for the ISOTROPIC material too, but only under REFINEMENT (64x40 -> 3.03e7, 96x60 ->
7.07e4).  Four near-degenerate eigenvalues on a shape with four corners is the tell.

Cause hypothesis: each perimeter node is offset along ITS OWN face normal, X = P[j] + zz*T*(-nin[j]).  At a
corner the two adjacent face normals differ by 90 deg, so the offset surfaces do not meet -- the corner
element is skewed and the wall is locally the wrong thickness.  Refining the perimeter shrinks the element
(a/nps) while the defect stays T/2, so the flaw grows RELATIVELY worse -- exactly the observed trend.

Fix: MITRED offset.  At node j use the angle bisector of the two adjacent edge normals, scaled by
1/(b.n) so the offset lands on the intersection of the two offset lines.  Interior nodes are unaffected
(b == n, scale 1), corners get the sqrt(2) miter.

Reports, for BOTH the old and mitred mesh: worst cell scaled-Jacobian, then the buckling eigenvalue, and
WHERE the mode lives (corner band vs face band).  Runs iso + aniso so the two failures are compared head on.
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
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ax = lambda x: S1 + (S2 - S1) * x / L


def sq_ring(a, n):
    nps = n // 4
    cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    P, Nin, Tg = [], [], []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        d = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-d[1], d[0]]); mid = 0.5 * (P0 + P1)
        if np.dot(-mid, nrm) < 0:
            nrm = -nrm
        for j in range(nps):
            P.append(P0 + (j / nps) * (P1 - P0)); Nin.append(nrm); Tg.append(d)
    return np.array(P), np.array(Nin), np.array(Tg)


def offsets(nin, mitre):
    """per-node offset vector (unit-thickness); mitred version closes the corner notch."""
    n = len(nin)
    if not mitre:
        return nin.copy()
    out = np.zeros_like(nin)
    for j in range(n):
        np_, nn = nin[(j - 1) % n], nin[j]
        b = np_ + nn; nb = np.linalg.norm(b)
        if nb < 1e-12:
            out[j] = nn; continue
        b /= nb
        out[j] = b / max(np.dot(b, nn), 1e-9)      # land on the intersection of the two offset lines
    return out


def ply_C(mat, ang_deg):
    E1, E2, G12 = mat["E1"], mat["E2"], mat["G12"]; nu12 = mat["nu12"]
    E3 = E2; G13 = G12; G23 = 0.4 * E2; nu13 = nu12; nu23 = 0.4
    nu21 = nu12 * E2 / E1; nu31 = nu13 * E3 / E1; nu32 = nu23 * E3 / E2
    S = np.array([[1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                  [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                  [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                  [0, 0, 0, 1 / G23, 0, 0], [0, 0, 0, 0, 1 / G13, 0], [0, 0, 0, 0, 0, 1 / G12]])
    C = np.linalg.inv(S)
    c, s = np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))
    Tm = np.array([[c*c, s*s, 0, 0, 0, 2*c*s], [s*s, c*c, 0, 0, 0, -2*c*s], [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, c, -s, 0], [0, 0, 0, s, c, 0], [-c*s, c*s, 0, 0, 0, c*c - s*s]])
    R = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tm) @ C @ R @ Tm @ np.linalg.inv(R)


def rot_global(Cl, e1, e2, e3):
    A = np.array([e1, e2, e3])
    K1 = A**2
    K2 = np.array([[A[i, 1]*A[i, 2], A[i, 2]*A[i, 0], A[i, 0]*A[i, 1]] for i in range(3)])
    K3 = np.array([[A[1, j]*A[2, j], A[2, j]*A[0, j], A[0, j]*A[1, j]] for j in range(3)]).T
    K4 = np.array([[A[1, (j+1) % 3]*A[2, (j+2) % 3] + A[1, (j+2) % 3]*A[2, (j+1) % 3],
                    A[2, (j+1) % 3]*A[0, (j+2) % 3] + A[2, (j+2) % 3]*A[0, (j+1) % 3],
                    A[0, (j+1) % 3]*A[1, (j+2) % 3] + A[0, (j+2) % 3]*A[1, (j+1) % 3]] for j in range(3)]).T
    Tt = np.block([[K1, 2 * K2], [K3, K4]])
    Rm = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tt) @ Cl @ Rm @ Tt @ np.linalg.inv(Rm)


def build(nper, nspan, nthk, mitre):
    xs = np.linspace(0.0, L, nspan + 1); zk = np.linspace(-0.5, 0.5, nthk + 1)
    X = np.zeros(((nspan + 1) * nper * (nthk + 1), 3))
    nid = lambda i, j, k: (i * nper + (j % nper)) * (nthk + 1) + k
    for i, x in enumerate(xs):
        P, nin, _ = sq_ring(ax(x), nper); off = offsets(nin, mitre)
        for j in range(nper):
            for k, zz in enumerate(zk):
                X[nid(i, j, k)] = (x, *(P[j] + (zz * T) * (-off[j])))
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(nspan) for j in range(nper) for k in range(nthk)], dtype=np.int64)
    return dmesh.create_mesh(MPI.COMM_WORLD, cells, X,
                             ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))), X, nid


def wall_thickness(X, nid, nper, nthk, i=0):
    """actual through-thickness distance at each perimeter station -- exposes the corner notch."""
    t = np.array([np.linalg.norm(X[nid(i, j, nthk)] - X[nid(i, j, 0)]) for j in range(nper)])
    return t


def solve(dom, aniso, nper, nthk, nspan, nmodes=6):
    if aniso:
        W = fem.functionspace(dom, basix.ufl.element("DG", dom.basix_cell(), 0, shape=(6, 6)))
        Cf = fem.Function(W); _, nin0, tg0 = sq_ring(1.0, nper); Cl = ply_C(MAT, -45.0)
        arr = Cf.x.array.reshape(-1, 36)
        for loc, og in enumerate(dom.topology.original_cell_index):
            j = (og % (nper * nthk)) // nthk
            arr[loc] = rot_global(Cl, np.array([1.0, 0, 0]), np.array([0.0, tg0[j][0], tg0[j][1]]),
                                  np.array([0.0, -nin0[j][0], -nin0[j][1]])).ravel()
        Cf.x.scatter_forward()
        vo = lambda e: ufl.as_vector([e[0, 0], e[1, 1], e[2, 2], 2*e[1, 2], 2*e[0, 2], 2*e[0, 1]])
        uv = lambda s: ufl.as_matrix([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
        ep_ = lambda v: ufl.sym(ufl.grad(v)); sg = lambda v: uv(ufl.dot(Cf, vo(ep_(v))))
    else:
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
    kg = -ufl.inner(sg(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
    K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
    KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
    ep = SLEPc.EPS().create(dom.comm); ep.setOperators(KG, K)
    ep.setProblemType(SLEPc.EPS.ProblemType.GHEP); ep.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    ep.setDimensions(nmodes, PETSc.DECIDE); ep.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    ep.setTolerances(1e-9, 1500); ep.solve()
    out = []
    Vx = V.tabulate_dof_coordinates()
    for i in range(ep.getConverged()):
        th = ep.getEigenvalue(i).real
        if not np.isfinite(th) or abs(th) < 1e-14:
            continue
        vr = fem.Function(V); ep.getEigenvector(i, vr.x.petsc_vec); vr.x.scatter_forward()
        vv = vr.x.array.reshape(-1, 3); mag = np.linalg.norm(vv[:, 1:], axis=1)
        mag = mag / (mag.max() + 1e-30)
        # corner-ness: distance in the y-z plane to the nearest of the 4 corner LINES (which taper)
        xx = Vx[:, 0]; aa = ax(np.clip(xx, 0, L))
        # near a CORNER means close to a/2 in BOTH y and z; min() would be 0 on every face point
        dcorner = np.maximum(np.abs(np.abs(Vx[:, 1]) - aa / 2), np.abs(np.abs(Vx[:, 2]) - aa / 2))
        near = dcorner < 0.05 * aa                      # within 5% of the side length of a corner
        frac = float(mag[near].mean() / (mag.mean() + 1e-30))
        out.append((abs(1.0 / th), ep.computeError(i), frac))
    out.sort(key=lambda r: r[0])
    return out


print("CORNER-NOTCH TEST: does the mitred offset remove the spurious degenerate modes?")
for (nper, nspan, nthk) in [(64, 40, 2), (96, 60, 2)]:
    for aniso in (False, True):
        print("\n=== nper=%d nspan=%d nthk=%d  %s ===" % (nper, nspan, nthk, "ANISO -45" if aniso else "ISO"))
        for mitre in (False, True):
            t0 = time.time()
            dom, X, nid = build(nper, nspan, nthk, mitre)
            tw = wall_thickness(X, nid, nper, nthk)
            res = solve(dom, aniso, nper, nthk, nspan)
            tag = "MITRED" if mitre else "as-built"
            print("   %-9s wall t: min=%.5f max=%.5f (nominal %.5f)   (%.0fs)"
                  % (tag, tw.min(), tw.max(), T, time.time() - t0))
            for (lam, err, frac) in res[:4]:
                print("        lam=%.5e  res=%.1e  corner_concentration=%.2f %s"
                      % (lam, err, frac, "<-- CORNER MODE" if frac > 1.5 else ""))
