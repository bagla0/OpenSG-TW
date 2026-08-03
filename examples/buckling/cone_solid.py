"""cone_solid.py -- the MISSING 3-D solid reference for the tapered circle, isotropic and m45.

Why this is the right next measurement.  The tapered SQUARE has a solid reference and sits 16% (iso) / 26%
(m45) below it.  The tapered CONE has only the classical equivalent-cylinder formula, which is itself
approximate at a semi-vertex angle of 14 deg, so it cannot resolve better than a few percent -- and there is
NO analytical anisotropic cone result at all, so the m45 cone accuracy is currently unmeasured.

A solid cone fixes both, and is trustworthy immediately: a cone has NO folds, so the corner-notch defect that
invalidated the as-built square reference cannot occur here.  Same FEniCSx machinery as the square.

Load and BCs are identical to the FSM setup so the comparison is apples to apples: unit TOTAL axial force on
the small end, left face fully fixed, right face free in x.
Material frame for m45 uses e1 = global x (the strip axis the FSM assumes), NOT the cone meridian, so the ply
angle means the same thing in both models.  The meridian is inclined 14 deg; using it instead would change
the effective fibre angle and break the comparison.
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
R1 = 1.0
# R2=R1 gives a PRISMATIC cylinder, needed to calibrate the shell-vs-solid offset before attributing
# anything on the cone to the taper (the tapered square has that calibration at 0.974; the circle did not).
R2 = float(os.environ.get("R2", "0.5"))
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
Rx = lambda x: R1 + (R2 - R1) * x / L


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


def run(nper, nspan, nthk, aniso, deg=2):
    xs = np.linspace(0.0, L, nspan + 1); zk = np.linspace(-0.5, 0.5, nthk + 1)
    th = np.linspace(0.0, 2 * np.pi, nper, endpoint=False)
    ct, st_ = np.cos(th), np.sin(th)
    X = np.zeros(((nspan + 1) * nper * (nthk + 1), 3))
    nid = lambda i, j, k: (i * nper + (j % nper)) * (nthk + 1) + k
    for i, x in enumerate(xs):
        R = Rx(x)
        for j in range(nper):
            for k, zz in enumerate(zk):
                r = R + zz * T                       # smooth contour: radial offset, no fold, no notch
                X[nid(i, j, k)] = (x, r * ct[j], r * st_[j])
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(nspan) for j in range(nper) for k in range(nthk)], dtype=np.int64)
    dom = dmesh.create_mesh(MPI.COMM_WORLD, cells, X,
                            ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))))
    if aniso:
        W = fem.functionspace(dom, basix.ufl.element("DG", dom.basix_cell(), 0, shape=(6, 6)))
        Cf = fem.Function(W); Cl = ply_C(MAT, -45.0); arr = Cf.x.array.reshape(-1, 36)
        for loc, og in enumerate(dom.topology.original_cell_index):
            j = (og % (nper * nthk)) // nthk
            e1 = np.array([1.0, 0.0, 0.0])                       # strip axis, matching the FSM convention
            e2 = np.array([0.0, -st_[j], ct[j]])                 # hoop tangent
            e3 = np.array([0.0, ct[j], st_[j]])                  # outward radial normal
            arr[loc] = rot_global(Cl, e1, e2, e3).ravel()
        Cf.x.scatter_forward()
        vo = lambda e: ufl.as_vector([e[0, 0], e[1, 1], e[2, 2], 2*e[1, 2], 2*e[0, 2], 2*e[0, 1]])
        uv = lambda s: ufl.as_matrix([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
        ep_ = lambda v: ufl.sym(ufl.grad(v)); sg = lambda v: uv(ufl.dot(Cf, vo(ep_(v))))
    else:
        mu = E / (2 * (1 + NU)); lm = E * NU / ((1 + NU) * (1 - 2 * NU))
        ep_ = lambda v: ufl.sym(ufl.grad(v))
        sg = lambda v: lm * ufl.tr(ep_(v)) * ufl.Identity(3) + 2 * mu * ep_(v)
    V = fem.functionspace(dom, ("Lagrange", deg, (3,)))
    du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    xg = dom.geometry.x; x0, x1 = xg[:, 0].min(), xg[:, 0].max(); fd = dom.topology.dim - 1
    lf = dmesh.locate_entities_boundary(dom, fd, lambda p: np.isclose(p[0], x0, atol=1e-9))
    rf = dmesh.locate_entities_boundary(dom, fd, lambda p: np.isclose(p[0], x1, atol=1e-9))
    z0 = fem.Constant(dom, 0.0)
    bcs = [fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(i), fd, lf), V.sub(i)) for i in range(3)]
    bcs += [fem.dirichletbc(z0, fem.locate_dofs_topological(V.sub(i), fd, rf), V.sub(i)) for i in (1, 2)]
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
    ep.setDimensions(6, PETSc.DECIDE); ep.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    ep.setTolerances(1e-9, 1500); ep.solve()
    ls = []
    for i in range(ep.getConverged()):
        t = ep.getEigenvalue(i).real
        if np.isfinite(t) and abs(t) > 1e-14:
            try:
                res = ep.computeError(i)
            except Exception:
                res = 0.0
            if res < 1e-4:                                   # reject unconverged/junk pairs
                ls.append(abs(1.0 / t))
    return (min(ls) if ls else np.nan), V.dofmap.index_map.size_global * 3


den = np.sqrt(3.0 * (1.0 - NU ** 2))
alpha = np.arctan((R1 - R2) / L)
P_cone = 2 * np.pi * E * T ** 2 * np.cos(alpha) ** 2 / den
if abs(R2 - R1) < 1e-12:                                      # prismatic cylinder calibration run
    FSM = {"iso": (2.950997e8, 2.950997e8), "m45": (3.685932e7, 3.685932e7)}
else:                                                          # tapered cone
    FSM = {"iso": (2.733177e8, 2.977014e8), "m45": (3.363872e7, 3.505277e7)}   # (per-station min, connected)

print("3-D SOLID reference for the TAPERED CONE  (no folds, so no corner-notch defect)")
print("   analytical (classical, equivalent-cylinder) iso cone P_cr = %.5e N\n" % P_cone)
print("   material  nper nspan nthk  |   ndof   | solid lam1     | per-station/solid  connected/solid  time")
for aniso, tag in ((False, "iso"), (True, "m45")):
    for (npe, nsp, nth) in [(64, 40, 2), (96, 60, 2)]:
        t0 = time.time()
        try:
            lam, nd = run(npe, nsp, nth, aniso)
        except Exception as ex:
            print("   %-8s %4d %5d %4d  | FAILED %s: %s" % (tag, npe, nsp, nth, type(ex).__name__, ex))
            continue
        per, con = FSM[tag]
        print("   %-8s %4d %5d %4d  | %8d | %.5e | %14.4f  %15.4f  %5.0fs"
              % (tag, npe, nsp, nth, nd, lam, per / lam, con / lam, time.time() - t0))
        if tag == "iso":
            print("            %sanalytical/solid = %.4f" % (" " * 30, P_cone / lam))
