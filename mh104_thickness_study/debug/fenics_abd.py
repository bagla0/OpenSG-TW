"""FEniCS per-layup plate ABD (6x6) for the mh104 f=0.2 layups at REAL E=10 Pa gelcoat, using the
exact ABD_mat from the reference mh104_0.2thickTimo_FEniCS.py.py.  Compare to jax_abd.py."""
import numpy as np
import dolfinx, ufl, basix
from mpi4py import MPI
from dolfinx.fem import form, petsc, functionspace
from ufl import SpatialCoordinate, as_tensor, Measure, TrialFunction, TestFunction, inner, lhs, rhs, dx, dot
from contextlib import ExitStack
import petsc4py.PETSc

material_parameters = [
    (1e1, 1e1, 1e1, 1e0, 1e0, 1e0, 0.3, 0.3, 0.3),                       # 0 gelcoat (REAL E=10 Pa)
    (1.03e10, 1.03e10, 1.03e10, 8e9, 8e9, 8e9, 0.3, 0.3, 0.3),           # 1 nexus
    (1.03e10, 1.03e10, 1.03e10, 8e9, 8e9, 8e9, 0.3, 0.3, 0.3),           # 2 db
    (3.7e10, 9e9, 9e9, 4e9, 4e9, 4e9, 0.28, 0.28, 0.28),                 # 3 ud
    (1e7, 1e7, 1e7, 2e5, 2e5, 2e5, 0.3, 0.3, 0.3)]                       # 4 balsa
F = 0.2
thick = [(F*0.000381, F*0.00051, F*0.00053*18),
         (F*0.000381, F*0.00051, F*0.00053*33),
         (F*0.000381, F*0.00051, F*0.00053*17, F*0.00053*38, F*0.003125, F*0.00053*37, F*0.00053*16),
         (F*0.000381, F*0.00051, F*0.00053*17, F*0.003125, F*0.00053*16),
         (F*0.00053*38, F*0.003125, F*0.00053*38)]
angle = [(0, 0, 20), (0, 0, 20), (0, 0, 20, 30, 0, 30, 20), (0, 0, 20, 0, 0), (0, 0, 0)]
matid = [(0, 1, 2), (0, 1, 2), (0, 1, 2, 3, 4, 3, 2), (0, 1, 2, 4, 2), (3, 4, 3)]
nlay = [len(a) for a in angle]
NAMES = ["layup_3(LE)", "layup_1(midFwd)", "layup_2(spar)", "layup_0(TE)", "layup_4(web)"]  # sub-index -> set in YAML


def ksp_solve(A, F_, V):
    w = dolfinx.fem.Function(V)
    ksp = petsc4py.PETSc.KSP().create(MPI.COMM_WORLD); ksp.setOperators(A)
    ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    ksp.setFromOptions(); ksp.solve(F_, w.vector)
    w.vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy(); return w


def nullspace(V):
    im = V.dofmap.index_map
    bas = [dolfinx.la.create_petsc_vector(im, V.dofmap.index_map_bs) for _ in range(3)]
    with ExitStack() as st:
        vl = [st.enter_context(b.localForm()) for b in bas]; arr = [np.asarray(b) for b in vl]
    dofs = [V.sub(i).dofmap.list for i in range(3)]
    for i in range(3):
        arr[i][dofs[i]] = 1.0
    dolfinx.la.orthonormalize(bas)
    return petsc4py.PETSc.NullSpace().create(bas, comm=MPI.COMM_WORLD)


def R_sig(C, t):
    th = np.deg2rad(t); c, s, cs = np.cos(th), np.sin(th), np.cos(th)*np.sin(th)
    R = np.array([(c**2, s**2, 0, 0, 0, -2*cs), (s**2, c**2, 0, 0, 0, 2*cs), (0, 0, 1, 0, 0, 0),
                  (0, 0, 0, c, s, 0), (0, 0, 0, -s, c, 0), (cs, -cs, 0, 0, 0, c**2-s**2)])
    return R @ C @ R.T


def ABD_mat(ii):
    elem = basix.ufl.element("Lagrange", "interval", 1, shape=(3,)); domain = ufl.Mesh(elem)
    th, s = [0], 0
    for k in thick[ii]:
        s -= k; th.append(s)
    pts = np.array(th).reshape(-1, 1) if False else np.array([[x, 0, 0] for x in th])
    cells = np.array([[k, k+1] for k in range(nlay[ii])])
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, pts, domain)
    nc = dom.topology.index_map(dom.topology.dim).size_local
    sd = dolfinx.mesh.meshtags(dom, dom.topology.dim, np.arange(nc, dtype=np.int32), np.arange(nc, dtype=np.int32))
    x = SpatialCoordinate(dom); dxm = Measure('dx')(domain=dom, subdomain_data=sd)
    Eps2 = as_tensor([(1, 0, 0, x[0], 0, 0), (0, 1, 0, 0, x[0], 0), (0, 0, 0, 0, 0, 0),
                      (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, x[0])])
    nph = len(np.arange(nc))

    def eps(v):
        E1 = ufl.as_vector([0, 0, v[2].dx(0), v[1].dx(0), v[0].dx(0), 0])
        return as_tensor([(E1[0], 0.5*E1[5], 0.5*E1[4]), (0.5*E1[5], E1[1], 0.5*E1[3]), (0.5*E1[4], 0.5*E1[3], E1[2])]), E1

    def C_of(i):
        E1, E2, E3, G12, G13, G23, v12, v13, v23 = material_parameters[matid[ii][i]]
        S = np.zeros((6, 6)); S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
        S[0, 1], S[0, 2] = -v12/E1, -v13/E1; S[1, 0], S[1, 2] = -v12/E1, -v23/E2
        S[2, 0], S[2, 1] = -v13/E1, -v23/E2; S[3, 3], S[4, 4], S[5, 5] = 1/G23, 1/G13, 1/G12
        return R_sig(np.linalg.inv(S), angle[ii][i])

    def sigma(v, i, Eps):
        C = C_of(i); s1 = dot(as_tensor(C), eps(v)[1]+Eps)
        return as_tensor([(s1[0], s1[5], s1[4]), (s1[5], s1[1], s1[3]), (s1[4], s1[3], s1[2])])

    V = functionspace(dom, basix.ufl.element("CG", "interval", 2, shape=(3,)))
    u, v = TrialFunction(V), TestFunction(V)
    F2 = sum([inner(sigma(u, i, Eps2[:, 0]), eps(v)[0])*dxm(i) for i in range(nph)])
    A = petsc.assemble_matrix(form(lhs(F2))); A.assemble(); null = nullspace(V); A.setNullSpace(null)
    xx = 3*V.dofmap.index_map.local_range[1]; V0 = np.zeros((xx, 6)); Dhe = np.zeros((xx, 6)); D_ee = np.zeros((6, 6))
    for p in range(6):
        F2 = sum([inner(sigma(u, i, Eps2[:, p]), eps(v)[0])*dxm(i) for i in range(nph)])
        Fv = petsc.assemble_vector(form(rhs(F2))); Fv.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(Fv); w = ksp_solve(A, Fv, V); Dhe[:, p] = Fv[:]; V0[:, p] = w.vector[:]
    D1 = V0.T @ (-Dhe)

    def Dee(i):
        C = C_of(i); x0 = x[0]
        return as_tensor([(C[0, 0], C[0, 1], C[0, 5], x0*C[0, 0], x0*C[0, 1], x0*C[0, 5]),
                          (C[1, 0], C[1, 1], C[1, 5], x0*C[1, 0], x0*C[1, 1], x0*C[1, 5]),
                          (C[5, 0], C[5, 1], C[5, 5], x0*C[5, 0], x0*C[5, 1], x0*C[5, 5]),
                          (x0*C[0, 0], x0*C[0, 1], x0*C[0, 5], x0*x0*C[0, 0], x0*x0*C[0, 1], x0*x0*C[0, 5]),
                          (x0*C[1, 0], x0*C[1, 1], x0*C[1, 5], x0*x0*C[1, 0], x0*x0*C[1, 1], x0*x0*C[1, 5]),
                          (x0*C[5, 0], x0*C[5, 1], x0*C[5, 5], x0*x0*C[5, 0], x0*x0*C[5, 1], x0*x0*C[5, 5])])
    for a in range(6):
        for b in range(6):
            D_ee[a, b] = dolfinx.fem.assemble_scalar(dolfinx.fem.form(sum([Dee(i)[a, b]*dxm(i) for i in range(nph)])))
    return D_ee + D1


for ii in range(5):
    abd = ABD_mat(ii)
    np.savetxt("/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/mh104_thickness_study/debug/abd_fenics_sub%d.txt" % ii, abd)
    print("=== sub %d  %s ===" % (ii, NAMES[ii]))
    print("  A11=%.6e A22=%.6e A66=%.6e  D11=%.6e D22=%.6e D66=%.6e  A16=%.4e B11=%.4e"
          % (abd[0, 0], abd[1, 1], abd[2, 2], abd[3, 3], abd[4, 4], abd[5, 5], abd[0, 2], abd[0, 3]), flush=True)
