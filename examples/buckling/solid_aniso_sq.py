"""solid_aniso_sq.py -- ANISOTROPIC [+-45]s tapered square: 3-D solid vs connected RM-FSM, same load.

LOAD (identical for both, and the same as the isotropic study):
    total axial force = 1.0 N, applied as a uniform pressure -1/A on the tip end face;
    root face u_x=0; BOTH end faces hold the cross-section (u_y=u_z=0)  [SS3-like, matching the FSM's
    sin(k_m x) basis which vanishes at both ends].  The solid solves statically -> sigma -> the membrane
    N11 = int sigma_xx dz is extracted per wall column and THAT SAME N11 drives the FSM.
    So lambda is directly the critical axial force in Newtons for both methods.

ANISOTROPY.  A [+-45]s laminate cannot be a single homogeneous solid: it is modelled as 4 PLY LAYERS through
the thickness (nthk=4, one element per ply), each with the orthotropic ply stiffness ROTATED about the WALL
NORMAL by its ply angle, expressed in the LOCAL WALL FRAME (e1 = span/x, e2 = contour tangent, e3 = normal)
and then pushed to global.  Stack order [+45,-45,-45,+45] from the outer surface inward.
The FSM uses the CLT ABD of the same stack, so both see the same laminate.
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

T, L = 0.02, 2.0
S1, S2 = 1.0, 0.5
NPER, NSPAN, NTHK = 64, 40, 2                     # single material -> no ply-conforming layers needed
PLIES = [-45.0]                                   # SINGLE -45 ply through the FULL thickness.
# Easier solid mesh (one homogeneous rotated-orthotropic material), and a STRONGER anisotropy test than a
# balanced [+-45]s stack: a single off-axis ply keeps BOTH A16/A26 (extension-shear) and D16/D26
# (bend-twist) coupling, whereas [+-45]s cancels the membrane pair.  B = 0 either way (homogeneous in z).
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
ax = lambda x: S1 + (S2 - S1) * x / L


def ply_C(mat, ang_deg):
    """3-D orthotropic ply stiffness (6x6 Voigt [11,22,33,23,13,12]) rotated by ang about the 3-axis."""
    E1, E2, G12 = mat["E1"], mat["E2"], mat["G12"]; nu12 = mat["nu12"]
    E3 = E2; G13 = G12; G23 = 0.4 * E2; nu13 = nu12; nu23 = 0.4
    nu21 = nu12 * E2 / E1; nu31 = nu13 * E3 / E1; nu32 = nu23 * E3 / E2
    S = np.array([[1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                  [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                  [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                  [0, 0, 0, 1 / G23, 0, 0],
                  [0, 0, 0, 0, 1 / G13, 0],
                  [0, 0, 0, 0, 0, 1 / G12]])
    C = np.linalg.inv(S)
    c, s = np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))
    # Voigt rotation about axis 3 (the wall normal); engineering-shear convention
    Tm = np.array([[c*c, s*s, 0, 0, 0, 2*c*s],
                   [s*s, c*c, 0, 0, 0, -2*c*s],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, c, -s, 0],
                   [0, 0, 0, s, c, 0],
                   [-c*s, c*s, 0, 0, 0, c*c - s*s]])
    R = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tm) @ C @ R @ Tm @ np.linalg.inv(R)


def rot_global(Cl, e1, e2, e3):
    """rotate a 6x6 Voigt stiffness from the local wall frame (e1,e2,e3) to global xyz."""
    A = np.array([e1, e2, e3])                      # rows = local axes in global coords
    K1 = A**2
    K2 = np.array([[A[i, 1]*A[i, 2], A[i, 2]*A[i, 0], A[i, 0]*A[i, 1]] for i in range(3)])
    K3 = np.array([[A[1, j]*A[2, j], A[2, j]*A[0, j], A[0, j]*A[1, j]] for j in range(3)]).T
    K4 = np.array([[A[1, (j+1) % 3]*A[2, (j+2) % 3] + A[1, (j+2) % 3]*A[2, (j+1) % 3],
                    A[2, (j+1) % 3]*A[0, (j+2) % 3] + A[2, (j+2) % 3]*A[0, (j+1) % 3],
                    A[0, (j+1) % 3]*A[1, (j+2) % 3] + A[0, (j+2) % 3]*A[1, (j+1) % 3]] for j in range(3)]).T
    Tt = np.block([[K1, 2 * K2], [K3, K4]])
    Rm = np.diag([1., 1., 1., 2., 2., 2.])
    Tinv = np.linalg.inv(Tt)
    return Tinv @ Cl @ Rm @ Tt @ np.linalg.inv(Rm)


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


def build():
    xs = np.linspace(0.0, L, NSPAN + 1); zk = np.linspace(-0.5, 0.5, NTHK + 1)
    X = np.zeros(((NSPAN + 1) * NPER * (NTHK + 1), 3))
    nid = lambda i, j, k: (i * NPER + (j % NPER)) * (NTHK + 1) + k
    for i, x in enumerate(xs):
        P, nin, _ = sq_ring(ax(x), NPER)
        for j in range(NPER):
            for k, zz in enumerate(zk):
                X[nid(i, j, k)] = (x, *(P[j] + (zz * T) * (-nin[j])))
    cells = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                      for i in range(NSPAN) for j in range(NPER) for k in range(NTHK)], dtype=np.int64)
    dom = dmesh.create_mesh(MPI.COMM_WORLD, cells, X,
                            ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))))
    return dom


print("ANISOTROPIC [+-45]s TAPERED SQUARE   (load: unit total axial force, SS3-like ends)")
dom = build()
# ---- per-cell global stiffness: ply angle by through-thickness index, wall frame by perimeter index ----
W = fem.functionspace(dom, basix.ufl.element("DG", dom.basix_cell(), 0, shape=(6, 6)))
Cf = fem.Function(W)
_, _, tg0 = sq_ring(1.0, NPER)
_, nin0, _ = sq_ring(1.0, NPER)
Cloc = [ply_C(MAT, a) for a in PLIES]
arr = Cf.x.array.reshape(-1, 36)
for loc, og in enumerate(dom.topology.original_cell_index):
    j = (og % (NPER * NTHK)) // NTHK; k = og % NTHK
    ply = min(int(k * len(PLIES) / NTHK), len(PLIES) - 1)   # maps through-thickness layer -> ply
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, tg0[j][0], tg0[j][1]])
    e3 = np.array([0.0, -nin0[j][0], -nin0[j][1]])
    arr[loc] = rot_global(Cloc[ply], e1, e2, e3).ravel()
Cf.x.scatter_forward()


def voigt(e):
    return ufl.as_vector([e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])


def unvoigt(s):
    return ufl.as_matrix([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


ep_ = lambda v: ufl.sym(ufl.grad(v))
sg = lambda v: unvoigt(ufl.dot(Cf, voigt(ep_(v))))
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
t0 = time.time()
LinearProblem(a_f, l_f, bcs=bcs, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                      "pc_factor_mat_solver_type": "mumps"}).solve()
S0 = fem.functionspace(dom, ("DG", 0)); sxx = fem.Function(S0)
sxx.interpolate(fem.Expression(sg(u0)[0, 0], S0.element.interpolation_points()))
N11 = np.zeros((NSPAN, NPER)); dz = T / NTHK
for loc, og in enumerate(dom.topology.original_cell_index):
    N11[og // (NPER * NTHK), (og % (NPER * NTHK)) // NTHK] += sxx.x.array[loc] * dz
kg = -ufl.inner(sg(u0), ufl.grad(du).T * ufl.grad(v)) * ufl.dx
K = assemble_matrix(fem.form(a_f), bcs=bcs, diagonal=1.0); K.assemble()
KG = assemble_matrix(fem.form(kg), bcs=bcs, diagonal=0.0); KG.assemble()
e2s = SLEPc.EPS().create(dom.comm); e2s.setOperators(KG, K)
e2s.setProblemType(SLEPc.EPS.ProblemType.GHEP); e2s.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
e2s.setDimensions(4, PETSc.DECIDE); e2s.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
e2s.setTolerances(1e-8, 600); e2s.solve()
# Filter by the ACTUAL residual: KRYLOVSCHUR/LARGEST_MAGNITUDE can return junk pairs, and blindly taking
# min(1/|theta|) then latches onto them (seen as lam ~ 5e4 instead of ~3e7, i.e. 600x low).  eps.computeError
# gives the relative residual ||KG x - theta K x|| / |theta|; a genuine mode has ~1e-10, junk is O(1).
ls = []
print("   raw spectrum (theta, lam=1/|theta|, residual):")
for i in range(e2s.getConverged()):
    th = e2s.getEigenvalue(i).real
    try:
        res = e2s.computeError(i)
    except Exception:
        res = np.nan
    lam_i = abs(1.0 / th) if abs(th) > 1e-14 else np.inf
    flag = "keep" if (np.isfinite(th) and abs(th) > 1e-14 and (not np.isfinite(res) or res < 1e-6)) else "DROP"
    print("     %2d  theta=%+.6e  lam=%.5e  res=%.2e  %s" % (i, th, lam_i, res, flag))
    if flag == "keep":
        ls.append(lam_i)
lam_solid = min(ls) if ls else np.nan
print("   SOLID [+-45]s lam1 = %.5e   (%.0fs)  end area A=%.4e" % (lam_solid, time.time() - t0, A))
print("   N11 range: %.4e .. %.4e" % (N11.min(), N11.max()))

# ---------------- FSM with the SAME N and the CLT ABD of the same stack ----------------
ABD = fsm.clt_abd([(a, T / len(PLIES)) for a in PLIES], MAT)   # same stack the solid was given
xmid = 0.5 * (np.linspace(0, L, NSPAN + 1)[:-1] + np.linspace(0, L, NSPAN + 1)[1:])


def mk(nsec):
    out = []
    for x in np.linspace(0, L, nsec):
        i = int(np.clip(np.searchsorted(xmid, x), 0, NSPAN - 1))
        P, _, _ = sq_ring(ax(x), NPER)
        st = np.array([[q, (q + 1) % len(P)] for q in range(len(P))])
        out.append((float(x), P, [ABD] * len(st), [np.array([N11[i, q], 0.0, 0.0]) for q in range(len(st))]))
    return out


per = []
for (x, P, Al, Nl) in mk(4):
    st = np.array([[q, (q + 1) % len(P)] for q in range(len(P))])
    lam = np.asarray(fsm.solve_fsm_multi(P, st, Al, Nl, L, 8, n_modes=2))
    per.append(float(lam[0]))
lam_c = np.asarray(fsm.solve_fsm_connected(mk(4), L, 8, n_modes=3, ngauss=32))
print("\n   per-station MIN (4 sec) = %.5e   /solid = %.4f" % (min(per), min(per) / lam_solid))
print("   CONNECTED      (4 sec) = %.5e   /solid = %.4f" % (lam_c[0], lam_c[0] / lam_solid))
print("   SOLID                  = %.5e" % lam_solid)
