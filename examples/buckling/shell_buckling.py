"""shell_buckling.py -- linear (Euler) buckling of a composite SHELL mesh (pathway B: shell + ABD).

Solve  (K + lambda*Kg) phi = 0  <=>  K phi = P * KGpos phi ,  P_cr = smallest positive eigenvalue.
  K      : shell material stiffness from the laminate ABD ([A B; B D]) + transverse shear Gs.
  KGpos  : geometric (stress) stiffness from the pre-buckling MEMBRANE resultants N=(Nxx,Nyy,Nxy),
           built from the UNIT reference load so P_cr is the buckling load factor.
Element: 4-node Mindlin-Reissner FACET shell, 2x2 Gauss membrane/bending + MITC4 assumed transverse
shear (tying points) -> no shear locking AND no spurious hourglass; local (u,v,w,thx,thy) transformed
to global 6-DOF with a small drilling penalty for the curved-shell assembly.  KG carries the membrane
prestress on all three local translational gradients (u,v,w) -- the (a)+(b) form that, on a flat facet,
is the discrete image of the curved-shell curvature geometric term; moment/transverse-shear/z^2 terms
vanish or are negligible at a CENTER (mid-surface) reference and are correctly omitted.  Sparse assembly
+ scipy.sparse.linalg.eigsh (regular mode, which='LA' on (-KG, M=K)).

Validated (see SHELL_BUCKLING_FORMULATION.md):
  * SS flat plate uniaxial:      N_cr = 0.9996 x (4 pi^2 D / a^2)   -- exact.
  * SS3 axial cylinder:          N_cr = 0.95-0.96 x  E t^2/(R sqrt(3(1-nu^2)))  -- classical benchmark.
  (A clamped-FREE cantilever cylinder gives ~0.38x from the free-edge boundary layer; that is CORRECT
   physics for a free end, not a code error -- the classical formula assumes SS/clamped BOTH ends.)

Run `python shell_buckling.py [plate|cyl|all]` for the self-tests.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

G2 = np.array([-1.0, 1.0]) / np.sqrt(3.0)                 # 2x2 Gauss points (weight 1 each)
_KDR_SCALE = 1e-3                                          # drilling-penalty scale (rel. to mean membrane diag)


def _shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]])   # (4,2): d/dxi, d/deta
    return N, dN


def _B_at(xy, xi, eta):
    """membrane Bm(3x20), bending Bb(3x20), shear Bs(2x20), w-grad Gw(2x20), detJ -- local DOF [u v w tx ty]."""
    N, dNr = _shape(xi, eta)
    J = dNr.T @ xy                                        # (2,2)
    detJ = np.linalg.det(J)
    dN = dNr @ np.linalg.inv(J).T                        # (4,2): d/dx, d/dy
    Bm = np.zeros((3, 20)); Bb = np.zeros((3, 20)); Bs = np.zeros((2, 20)); Gw = np.zeros((2, 20))
    Bm[0, 0::5] = dN[:, 0]; Bm[1, 1::5] = dN[:, 1]; Bm[2, 0::5] = dN[:, 1]; Bm[2, 1::5] = dN[:, 0]
    Bb[0, 3::5] = dN[:, 0]; Bb[1, 4::5] = dN[:, 1]; Bb[2, 3::5] = dN[:, 1]; Bb[2, 4::5] = dN[:, 0]
    Bs[0, 2::5] = dN[:, 0]; Bs[0, 3::5] = N              # gamma_xz = w,x + beta_x  (rot dof 3; matches kappa_xx=beta_x,x)
    Bs[1, 2::5] = dN[:, 1]; Bs[1, 4::5] = N              # gamma_yz = w,y + beta_y  (rot dof 4; matches kappa_yy=beta_y,y)
    Gw[0, 2::5] = dN[:, 0]; Gw[1, 2::5] = dN[:, 1]       # w,x ; w,y
    return Bm, Bb, Bs, Gw, detJ, J, dN


def element_K_KG(xyl, ABD, Gs, Nvec):
    """local 20x20 Ke, KGe for a flat quad. xyl (4,2) local in-plane coords; ABD (6,6); Gs (2,2); Nvec (3,).
    Transverse shear via MITC4 (assumed natural strain) -> no shear locking AND no spurious hourglass."""
    A, B, D = ABD[:3, :3], ABD[:3, 3:], ABD[3:, 3:]
    Nm = np.array([[Nvec[0], Nvec[2]], [Nvec[2], Nvec[1]]])
    Ke = np.zeros((20, 20)); KGe = np.zeros((20, 20))
    # MITC4 tying points: covariant gamma_xi at A(0,-1)&C(0,1); gamma_eta at B(1,0)&D(-1,0)
    _, _, BsA, _, _, JA, _ = _B_at(xyl, 0.0, -1.0); gxiA = (JA @ BsA)[0]
    _, _, BsC, _, _, JC, _ = _B_at(xyl, 0.0,  1.0); gxiC = (JC @ BsC)[0]
    _, _, BsB, _, _, JB, _ = _B_at(xyl, 1.0,  0.0); getB = (JB @ BsB)[1]
    _, _, BsD, _, _, JD, _ = _B_at(xyl, -1.0, 0.0); getD = (JD @ BsD)[1]
    for xi in G2:
        for eta in G2:
            Bm, Bb, _, Gw, detJ, Jg, dN = _B_at(xyl, xi, eta)
            Ke += (Bm.T @ A @ Bm + Bm.T @ B @ Bb + Bb.T @ B.T @ Bm + Bb.T @ D @ Bb) * detJ
            # FULL geometric stiffness: sum over gradients of ALL THREE local displacements.
            # (For a flat plate the u,v terms drop out under the in-plane BC; for a CURVED shell
            #  they carry the curvature-membrane coupling that a w-only KG misses.)
            Gx = np.zeros((2, 20)); Gx[0, 0::5] = dN[:, 0]; Gx[1, 0::5] = dN[:, 1]
            Gy = np.zeros((2, 20)); Gy[0, 1::5] = dN[:, 0]; Gy[1, 1::5] = dN[:, 1]
            KGe += (Gx.T @ Nm @ Gx + Gy.T @ Nm @ Gy + Gw.T @ Nm @ Gw) * detJ
            Bgxi = 0.5 * (1 - eta) * gxiA + 0.5 * (1 + eta) * gxiC        # interpolate covariant shear
            Bget = 0.5 * (1 + xi) * getB + 0.5 * (1 - xi) * getD
            Bsc = np.linalg.inv(Jg) @ np.vstack([Bgxi, Bget])            # -> Cartesian assumed shear
            Ke += (Bsc.T @ Gs @ Bsc) * detJ
    return Ke, KGe


def _iso_ABD(E, nu, h):
    C = E / (1 - nu * nu)
    Qm = C * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    ABD = np.zeros((6, 6)); ABD[:3, :3] = Qm * h; ABD[3:, 3:] = Qm * h ** 3 / 12.0
    Gs = (5.0 / 6.0) * (E / (2 * (1 + nu))) * h * np.eye(2)
    return ABD, Gs


def solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec_e, fixed_dof, n_modes=6):
    """Global GEP.  nodes (nn,3); quads (ne,4); per-element ABD_e (ne,6,6),Gs_e (ne,2,2),Nvec_e (ne,3);
    fixed_dof = list of constrained global DOF (6/node).  Returns (loads[n_modes], modes[nn,6,n_modes])."""
    nn = len(nodes); ndof = 6 * nn; ne = len(quads)
    flat = _flat_node_mask(nodes, quads)                        # drilling spring only at coplanar nodes (skip folds)
    KD = np.zeros((ne, 24, 24)); KGD = np.zeros((ne, 24, 24)); GDOF = np.zeros((ne, 24), int)
    for e, q in enumerate(quads):
        T, xyl = _elem_frame(nodes, q)
        Ke, KGe = element_K_KG(xyl, ABD_e[e], Gs_e[e], Nvec_e[e])
        L = _L_lg(T)                                            # local 20 <- global 24
        Kd = L.T @ Ke @ L; KGd = L.T @ KGe @ L
        kdr = _KDR_SCALE * np.mean(np.abs(np.diag(Ke)[2::5]))    # small drilling penalty
        Kn = kdr * np.outer(T[2], T[2])                          # about the ELEMENT NORMAL (flat facet e3=z ->
        for a in range(4):                                      # old theta_z term); NOT at folds (leaks to bending)
            if flat[q[a]]:
                Kd[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] += Kn
        KD[e] = Kd; KGD[e] = KGd
        GDOF[e] = np.concatenate([np.arange(6 * n, 6 * n + 6) for n in q])
    I = np.broadcast_to(GDOF[:, :, None], (ne, 24, 24)).ravel()  # vectorized COO assembly
    J = np.broadcast_to(GDOF[:, None, :], (ne, 24, 24)).ravel()
    K = sp.coo_matrix((KD.ravel(), (I, J)), shape=(ndof, ndof)).tocsr()
    KG = sp.coo_matrix((KGD.ravel(), (I, J)), shape=(ndof, ndof)).tocsr()
    free = np.setdiff1d(np.arange(ndof), np.asarray(fixed_dof, int))
    Kff = K[free][:, free]; KGff = KG[free][:, free]
    # (K + P KG) phi = 0 -> solve (-KG) phi = (1/P) K phi and take the LARGEST 1/P (= smallest buckling P).
    # KG is singular (w-only), so we make K the (SPD) mass matrix and avoid shift-inverting the singular one.
    w, vecs = eigsh((-KGff).tocsc(), k=n_modes, M=Kff.tocsc(), which="LA")
    keep = w > 1e-12                                       # drop the KG-null (1/P ~ 0, i.e. P -> inf) modes
    w = w[keep]; vecs = vecs[:, keep]
    vals = 1.0 / w
    order = np.argsort(vals); vals = vals[order]; vecs = vecs[:, order]
    nret = len(vals)
    modes = np.zeros((nn, 6, nret)); full = np.zeros(ndof)
    for m in range(nret):
        full[:] = 0; full[free] = vecs[:, m]
        modes[:, :, m] = full.reshape(nn, 6)
    return vals, modes


def _flat_node_mask(nodes, quads, cos_thresh=0.5):
    """True at nodes whose incident element normals are ~coplanar -> safe to add a drilling spring.
    False at FOLDS/corners (dihedral > ~60 deg), where the junction geometry already couples the
    drilling rotation into the neighbour wall's bending and removes the singularity; adding a spring
    there would LEAK into that wall's PHYSICAL bending rotation and over-stiffen the fold (a square-tube
    corner clamps toward k=6.97 instead of the correct k=4).  |dot| is winding-insensitive."""
    ne = len(quads); nrm = np.zeros((ne, 3))
    for e, q in enumerate(quads):
        T, _ = _elem_frame(nodes, q); nrm[e] = T[2]
    acc = [[] for _ in range(len(nodes))]
    for e, q in enumerate(quads):
        for n in q:
            acc[n].append(e)
    flat = np.ones(len(nodes), bool)
    for n, es in enumerate(acc):
        if len(es) > 1:
            N = nrm[es]
            if np.abs(N @ N.T).min() < cos_thresh:
                flat[n] = False
    return flat


def _elem_frame(nodes, q):
    """local orthonormal frame T (rows e1,e2,e3) and in-plane coords xyl (4,2) for a quad."""
    X = nodes[q]
    v1 = X[1] - X[0]; v2 = X[3] - X[0]
    e3 = np.cross(v1, v2); e3 /= np.linalg.norm(e3) + 1e-30
    e1 = v1 / (np.linalg.norm(v1) + 1e-30); e2 = np.cross(e3, e1)
    T = np.array([e1, e2, e3])
    return T, ((X - X[0]) @ T.T)[:, :2]


def _L_lg(T):
    """local-20 (u v w tx ty) <- global-24 (ux uy uz rx ry rz) transform: u_loc = L u_glob."""
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3:5 * a + 5, 6 * a + 3:6 * a + 6] = T[:2]
    return L


def assemble_K(nodes, quads, ABD_e, Gs_e):
    """Global material stiffness K (6/node), same element + drilling as solve_buckling (KG omitted)."""
    nn = len(nodes); ndof = 6 * nn; ne = len(quads)
    z3 = np.zeros(3)
    flat = _flat_node_mask(nodes, quads)                        # drilling spring only at coplanar nodes (skip folds)
    KD = np.zeros((ne, 24, 24)); GDOF = np.zeros((ne, 24), int)
    for e, q in enumerate(quads):
        T, xyl = _elem_frame(nodes, q)
        Ke, _ = element_K_KG(xyl, ABD_e[e], Gs_e[e], z3)         # Nvec=0 -> KG unused
        L = _L_lg(T); Kd = L.T @ Ke @ L
        kdr = _KDR_SCALE * np.mean(np.abs(np.diag(Ke)[2::5]))
        Kn = kdr * np.outer(T[2], T[2])                          # drilling penalty about the element normal
        for a in range(4):
            if flat[q[a]]:                                      # NOT at folds (would leak into wall bending)
                Kd[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] += Kn
        KD[e] = Kd; GDOF[e] = np.concatenate([np.arange(6 * n, 6 * n + 6) for n in q])
    I = np.broadcast_to(GDOF[:, :, None], (ne, 24, 24)).ravel()
    J = np.broadcast_to(GDOF[:, None, :], (ne, 24, 24)).ravel()
    return sp.coo_matrix((KD.ravel(), (I, J)), shape=(ndof, ndof)).tocsr()


def solve_static(nodes, quads, ABD_e, Gs_e, fext, fixed_dof, K=None):
    """Linear static solve K u = fext with fixed DOF constrained to zero.  fext (ndof,) global load vector."""
    from scipy.sparse.linalg import spsolve
    if K is None:
        K = assemble_K(nodes, quads, ABD_e, Gs_e)
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), np.asarray(fixed_dof, int))
    u = np.zeros(ndof)
    u[free] = spsolve(K[free][:, free].tocsc(), np.asarray(fext, float)[free])
    return u


def element_membrane_N(nodes, quads, ABD_e, u):
    """per-element membrane stress resultant N=[Nxx,Nyy,Nxy] (LOCAL frame) from a global displacement u.
    N = A eps + B kappa with the element-averaged membrane strain eps and curvature kappa."""
    Ne = np.zeros((len(quads), 3))
    for e, q in enumerate(quads):
        T, xyl = _elem_frame(nodes, q); L = _L_lg(T)
        ug = np.concatenate([u[6 * n:6 * n + 6] for n in q])     # (24,) global elem dof
        ul = L @ ug                                              # (20,) local
        A = ABD_e[e][:3, :3]; B = ABD_e[e][:3, 3:]
        eps = np.zeros(3); kap = np.zeros(3); ar = 0.0
        for xi in G2:
            for eta in G2:
                Bm, Bb, _, _, detJ, _, _ = _B_at(xyl, xi, eta)
                eps += (Bm @ ul) * detJ; kap += (Bb @ ul) * detJ; ar += detJ
        Ne[e] = A @ (eps / ar) + B @ (kap / ar)
    return Ne


def validate_plate(nx=24, a=1.0, h=0.01, E=200e9, nu=0.3):
    """Simply-supported square plate, uniform uniaxial compression Nx=-1: expect N_cr = 4 pi^2 D / a^2."""
    xs = np.linspace(0, a, nx + 1); ys = np.linspace(0, a, nx + 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    idx = lambda i, j: i * (nx + 1) + j
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nx) for j in range(nx)])
    ABD, Gs = _iso_ABD(E, nu, h)
    ne = len(quads)
    ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec_e = np.repeat(np.array([-1.0, 0.0, 0.0])[None], ne, 0)     # unit x-compression
    # BC: pin in-plane (u,v) everywhere (prescribed membrane state) + w=0 on 4 edges (SS)
    fixed = []
    nn = len(nodes)
    for n in range(nn):
        fixed += [6 * n + 0, 6 * n + 1]                            # u, v
        fixed += [6 * n + 5]                                       # drilling
    for i in range(nx + 1):
        for j in range(nx + 1):
            if i in (0, nx) or j in (0, nx):
                fixed += [6 * idx(i, j) + 2]                       # w on edges
    fixed = np.unique(fixed)
    loads, _ = solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec_e, fixed, n_modes=4)
    D = E * h ** 3 / (12 * (1 - nu ** 2))
    Ncr_analytic = 4 * np.pi ** 2 * D / a ** 2
    print("SS plate uniaxial buckling  (nx=%d)" % nx)
    print("  N_cr  FE       = %.4f  N/m" % loads[0])
    print("  N_cr  analytic = %.4f  N/m  (4 pi^2 D / a^2, D=%.4e)" % (Ncr_analytic, D))
    print("  ratio FE/analytic = %.4f   (first 4 loads: %s)" % (loads[0] / Ncr_analytic, np.round(loads[:4], 3)))
    return loads[0], Ncr_analytic


def _cyl_mode_n(mode, nc, nl, R):
    """Dominant circumferential wave number n of a cylinder buckling mode (FFT of radial disp, mid ring)."""
    i = nl // 2
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    ur = np.array([mode[i * nc + j, 1] * np.cos(th[j]) + mode[i * nc + j, 2] * np.sin(th[j])
                   for j in range(nc)])                        # radial component uy*cos + uz*sin
    return int(np.argmax(np.abs(np.fft.rfft(ur))[1:]) + 1)     # skip DC


def validate_cylinder(mesh=(160, 80), R=1.0, L=2.0, t=0.02, E=200e9, nu=0.3, bc="SS", verbose=True):
    """Axially-compressed thin cylinder -- the CLASSICAL benchmark:  N_cr = E t^2 / (R sqrt(3(1-nu^2))).

    bc='SS' = SS3 (radial w=0 AND tangential v=0 at BOTH end rings, axial+rotations free) -- the BC the
    classical formula assumes; expect ratio ~0.95-1.05.  bc='clamp-free' = cantilever (blade BC); its
    ~0.38x is the free-edge boundary layer, correct physics for a free end (do NOT compare to classical)."""
    nc, nl = mesh
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    xs = np.linspace(0, L, nl + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
    idx = lambda i, j: i * nc + (j % nc)                       # wrap circumferentially -> closed cylinder
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nl) for j in range(nc)])   # edge 0->1 = axial (local e1)
    ABD, Gs = _iso_ABD(E, nu, t)
    ne = len(quads)
    ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec_e = np.repeat(np.array([-1.0, 0.0, 0.0])[None], ne, 0)  # unit axial compression (local Nxx)
    fixed = []
    for j in range(nc):
        r0, rL = idx(0, j), idx(nl, j)
        if bc == "SS":                                         # SS3: radial+tangential (uy,uz) fixed both ends
            fixed += [6 * r0 + 1, 6 * r0 + 2, 6 * r0 + 0, 6 * rL + 1, 6 * rL + 2]   # axial anchored at root
        elif bc == "clamp-free":
            fixed += [6 * r0 + k for k in range(6)]            # clamp root ring; tip free (cantilever)
    fixed = np.unique(fixed)
    loads, modes = solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec_e, fixed, n_modes=6)
    Ncr = E * t ** 2 / (R * np.sqrt(3 * (1 - nu ** 2)))
    n1 = _cyl_mode_n(modes[:, :, 0], nc, nl, R)
    if verbose:
        Z = L ** 2 / (R * t) * np.sqrt(1 - nu ** 2)
        print("cylinder axial buckling  (nc=%d nl=%d, R/t=%.0f, L/R=%.1f, Batdorf Z=%.1f, bc=%s)"
              % (nc, nl, R / t, L / R, Z, bc))
        print("  N_cr  FE        = %.4e  N/m   (first 4: %s)" % (loads[0], np.array2string(loads[:4], precision=3)))
        print("  N_cr  classical = %.4e  N/m   (E t^2 / (R sqrt(3(1-nu^2))))" % Ncr)
        print("  ratio FE/classical = %.4f   circumferential mode n = %d" % (loads[0] / Ncr, n1))
    return loads[0], Ncr


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "plate"
    if which in ("plate", "all"):
        validate_plate()
    if which in ("cyl", "cylinder", "all"):
        validate_cylinder(mesh=(160, 80), bc="SS")
    if which in ("conv", "all"):                              # facet convergence toward classical
        print("\nSS cylinder mesh convergence (ratio should be stable, ~0.95-1.0):")
        prev = None
        for m in [(120, 60), (160, 80), (240, 120)]:
            r = validate_cylinder(mesh=m, bc="SS", verbose=False)
            rel = "" if prev is None else "  (d=%.2f%%)" % (100 * abs(r[0] - prev) / prev)
            print("  nc=%3d nl=%3d : ratio=%.4f%s" % (m[0], m[1], r[0] / r[1], rel)); prev = r[0]
        cf = validate_cylinder(mesh=(160, 80), bc="clamp-free", verbose=False)
        print("  clamp-free cantilever (blade BC): ratio=%.3f  (free-edge band 0.3-0.5, NOT vs classical)"
              % (cf[0] / cf[1]))
