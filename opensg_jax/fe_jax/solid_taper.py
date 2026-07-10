"""solid_taper.py -- JAX 3-D SOLID tapered-segment MSG homogenization (Timoshenko 6x6).

The FEniCS-OpenSG algorithm (opensg.core.solid.compute_stiffness(Taper=True) +
compute_timo_boun), reimplemented in JAX with ELEMENT-TYPE BATCHES (the fea-in-jax
pattern): elements of each cell type (hex8 / tet4 in the volume, quad4 / tri3 on the
extracted boundary) form one vmapped batch; every batch's COO triplets concatenate
under the single global dof numbering -- so MIXED hex+tet (and quad+tri boundary)
meshes assemble natively into one system.

Pipeline (identical math to the FEniCS solid):
  1. read the 3-D solid segment YAML (string/1-based "x y z" nodes, z = beam axis ->
     permuted beam-first; 8-node hex and/or 4-node tet elements; per-element material
     set + 9-float orientation frame permuted to beam-first);
  2. EXTRACT the L/R boundary submeshes: parent faces whose nodes all lie on the end
     plane (hex face -> quad4, tet face -> tri3; mixed allowed), inheriting the parent
     element's material + frame -- this is the JAX analogue of dolfinx create_submesh;
  3. solve each boundary 2-D SG (gamma_h dim=2, 4-mode nullspace KKT: 3 translations +
     twist) -> boundary Timoshenko 6x6 + fluctuations V0_b, V1_b;
  4. solve the 3-D segment (gamma_h dim=3) with DIRICHLET BCs = the boundary
     fluctuations scattered through the node2seg maps; 4 V0 cases + 4 V1 cases reuse
     ONE SuperLU factorization (never pypardiso: silently wrong on webbed T-junction
     reduced blocks -- see mitc_rm_segment/segment_element.py);
  5. Timoshenko finalize (B_tim / C_tim / Q_tim -> 6x6 reorder), literally the FEniCS
     block algebra, with the segment /L scaling.

Voigt order [11, 22, 33, 23, 13, 12] with axis 0 = beam, matching opensg.utils.solid.
"""
import time

import numpy as np
import yaml
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from opensg_jax.fe_jax.solid_timo import build_orthotropic_C, rotate_C_with_matrix


# ================================================================================ reader
_PERM3 = [2, 0, 1]                              # (x, y, z=beam) -> (beam, x, y)
_PERMF = [2, 0, 1, 5, 3, 4, 8, 6, 7]            # same permutation per frame vector


def read_solid_segment_yaml(path):
    """OpenSG 3-D solid segment YAML -> dict(nodes (N,3) BEAM-FIRST, batches, mat_param).

    batches = {"hex8": (conn (n,8) 0-based, mat_id (n,), frames (n,9))} and/or "tet4".
    Mixed hex+tet segments are supported (each type is its own batch)."""
    CL = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    d = yaml.load(open(path), Loader=CL)
    nodes = np.array([[float(v) for v in r[0].split()] for r in d["nodes"]])[:, _PERM3]
    conns = [[int(v) - 1 for v in r[0].split()] for r in d["elements"]]
    nelem = len(conns)

    mat_of = np.zeros(nelem, int)
    mat_param, names = [], []
    for mi, m in enumerate(d["materials"]):
        e = m.get("elastic", m)                            # flat E/G/nu or elastic-nested
        E, G, nu = e["E"], e["G"], e["nu"]
        mat_param.append([E[0], E[1], E[2], G[0], G[1], G[2], nu[0], nu[1], nu[2]])
        names.append(m["name"])
    name_ix = {n: i for i, n in enumerate(names)}
    for s in d["sets"]["element"]:
        mi = name_ix[s["name"]]
        for lab in s["labels"]:
            mat_of[lab - 1] = mi

    frames = np.array([[float(v) for v in (r if not isinstance(r[0], str) else r[0].split())]
                       for r in d["elementOrientations"]])[:, _PERMF]

    batches = {}
    for key, nn in (("tet4", 4), ("hex8", 8)):
        ix = [k for k, c in enumerate(conns) if len(c) == nn]
        if ix:
            batches[key] = (np.array([conns[k] for k in ix], int),
                            mat_of[ix], frames[ix])
    return dict(nodes=nodes, batches=batches, mat_param=np.array(mat_param),
                nelem=nelem)


# ================================================================ boundary submesh extraction
_HEXF = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
_TETF = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def extract_boundary_submesh(seg, side):
    """The 2-D boundary cross-section of the 3-D segment at the `side` ('L'|'R') end:
    faces of parent cells whose nodes ALL lie on the end plane (hex->quad4, tet->tri3,
    mixed allowed), inheriting the parent's material + frame.  Returns
    dict(nodes (M,3) beam-first, batches2d, node2seg (M,), ...) -- the JAX analogue of
    the dolfinx create_submesh used by FEniCS-OpenSG."""
    nodes = seg["nodes"]
    b = nodes[:, 0]
    zend = b.min() if side == "L" else b.max()
    tol = 1e-6 * max(b.max() - b.min(), 1.0)
    on = np.abs(b - zend) < tol

    faces, fmat, ffrm = {"quad4": [], "tri3": []}, {"quad4": [], "tri3": []}, {"quad4": [], "tri3": []}
    for key, FACES, fkey in (("hex8", _HEXF, "quad4"), ("tet4", _TETF, "tri3")):
        if key not in seg["batches"]:
            continue
        conn, mat, frm = seg["batches"][key]
        for e in range(len(conn)):
            for f in FACES:
                fn = conn[e][list(f)]
                if on[fn].all():
                    faces[fkey].append(fn)
                    fmat[fkey].append(mat[e])
                    ffrm[fkey].append(frm[e])
    used = np.unique(np.concatenate([np.ravel(np.array(faces[k], int))
                                     for k in faces if faces[k]]))
    g2l = {int(g): i for i, g in enumerate(used)}
    batches2d = {}
    for k in ("quad4", "tri3"):
        if faces[k]:
            conn2 = np.array([[g2l[int(n)] for n in fn] for fn in faces[k]], int)
            batches2d[k] = (conn2, np.array(fmat[k], int), np.array(ffrm[k]))
    return dict(nodes=nodes[used], batches=batches2d, node2seg=used,
                mat_param=seg["mat_param"], zend=zend)


# ============================================================================ element kernels
_S8 = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
_G2 = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _rule(kind):
    """(xi (q,dim), w (q,), shapeN fn, dNdxi fn) for each cell type."""
    if kind == "hex8":
        pts = np.array([(a, b, c) for a in _G2 for b in _G2 for c in _G2])
        w = np.ones(8)

        def N(x):
            return 0.125 * (1 + x[0] * _S8[:, 0]) * (1 + x[1] * _S8[:, 1]) * (1 + x[2] * _S8[:, 2])

        def dN(x):
            return np.stack([
                0.125 * _S8[:, 0] * (1 + x[1] * _S8[:, 1]) * (1 + x[2] * _S8[:, 2]),
                0.125 * _S8[:, 1] * (1 + x[0] * _S8[:, 0]) * (1 + x[2] * _S8[:, 2]),
                0.125 * _S8[:, 2] * (1 + x[0] * _S8[:, 0]) * (1 + x[1] * _S8[:, 1])], 0)
        return pts, w, N, dN
    if kind == "tet4":
        a, b = 0.5854101966249685, 0.1381966011250105
        pts = np.array([(b, b, b), (a, b, b), (b, a, b), (b, b, a)])
        w = np.full(4, 1.0 / 24.0)

        def N(x):
            return np.array([1 - x[0] - x[1] - x[2], x[0], x[1], x[2]])

        def dN(x):
            return np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]], float)
        return pts, w, N, dN
    if kind == "quad4":
        pts = np.array([(a, b) for a in _G2 for b in _G2])
        w = np.ones(4)
        s = _S8[:4, :2]

        def N(x):
            return 0.25 * (1 + x[0] * s[:, 0]) * (1 + x[1] * s[:, 1])

        def dN(x):
            return np.stack([0.25 * s[:, 0] * (1 + x[1] * s[:, 1]),
                             0.25 * s[:, 1] * (1 + x[0] * s[:, 0])], 0)
        return pts, w, N, dN
    if kind == "tri3":
        pts = np.array([(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)])
        w = np.full(3, 1.0 / 6.0)

        def N(x):
            return np.array([1 - x[0] - x[1], x[0], x[1]])

        def dN(x):
            return np.array([[-1, 1, 0], [-1, 0, 1]], float)
        return pts, w, N, dN
    raise KeyError(kind)


def _tables(kind):
    pts, w, N, dN = _rule(kind)
    return (jnp.asarray(np.stack([N(p) for p in pts])),          # (q, nn)
            jnp.asarray(np.stack([dN(p) for p in pts])),          # (q, dim, nn)
            jnp.asarray(w))


def _gamma_e(xq):
    """Gamma_e (6,4) at point xq=(x1,x2,x3): FEniCS opensg.utils.solid.gamma_e."""
    z = jnp.zeros(())
    o = jnp.ones(())
    return jnp.array([[o, z, xq[2], -xq[1]],
                      [z, z, z, z], [z, z, z, z], [z, z, z, z],
                      [z, xq[1], z, z],
                      [z, -xq[2], z, z]])


def _Bh(grads, nn, dim3):
    """gamma_h B-matrix (6, 3nn) from per-node gradients grads (3, nn) in BEAM-FIRST axes.
    dim3=True -> 3-D operator (beam derivative rows on); False -> boundary dim=2 operator
    (row0 = 0, no beam-derivative terms) per opensg.utils.solid.gamma_h."""
    B = jnp.zeros((6, 3 * nn))
    i = jnp.arange(nn)
    g0, g1, g2 = grads[0], grads[1], grads[2]
    if dim3:
        B = B.at[0, 3 * i + 0].set(g0)
        B = B.at[4, 3 * i + 0].add(g2).at[4, 3 * i + 2].add(g0)
        B = B.at[5, 3 * i + 0].add(g1).at[5, 3 * i + 1].add(g0)
    else:
        B = B.at[4, 3 * i + 0].set(g2)
        B = B.at[5, 3 * i + 0].set(g1)
    B = B.at[1, 3 * i + 1].set(g1)
    B = B.at[2, 3 * i + 2].set(g2)
    B = B.at[3, 3 * i + 1].add(g2).at[3, 3 * i + 2].add(g1)
    return B


def _Bl(shp, nn):
    """gamma_l B-matrix (6, 3nn) from shape VALUES shp (nn,): [v0,0,0,0,v2,v1]."""
    B = jnp.zeros((6, 3 * nn))
    i = jnp.arange(nn)
    B = B.at[0, 3 * i + 0].set(shp)
    B = B.at[4, 3 * i + 2].set(shp)
    B = B.at[5, 3 * i + 1].set(shp)
    return B


def _make_kernel(kind, dim3):
    """JIT element kernel: (xe (nn,3) beam-first coords, C (6,6)) ->
    (Khh, Kll, Khl(nn3,nn3), Khe, Kle(nn3,4), Dee(4,4))  [Khe stored WITH the minus]."""
    Nq, dNq, Wq = _tables(kind)
    nn = Nq.shape[1]
    volume = kind in ("hex8", "tet4")

    def kernel(xe, C):
        def qp(q):
            dN = dNq[q]                                   # (dim, nn)
            if volume:
                J = dN @ xe                               # (3,3)
                detJ = jnp.abs(jnp.linalg.det(J))
                g = jnp.linalg.solve(J, dN)               # (3, nn) grads wrt (beam,x2,x3)
            else:
                J = dN @ xe[:, 1:3]                       # (2,2) in-plane
                detJ = jnp.abs(jnp.linalg.det(J))
                g2 = jnp.linalg.solve(J, dN)              # (2, nn) grads wrt (x2,x3)
                g = jnp.concatenate([jnp.zeros((1, nn)), g2], 0)
            xq = Nq[q] @ xe                               # (3,)
            Bh = _Bh(g, nn, dim3)
            Bl = _Bl(Nq[q], nn)
            Ge = _gamma_e(xq)
            wd = Wq[q] * detJ
            CBh = C @ Bh
            CGe = C @ Ge
            return (wd * Bh.T @ CBh,                      # Khh
                    wd * Bl.T @ (C @ Bl),                 # Kll
                    wd * Bl.T @ CBh,                      # Khl  rows=gamma_l, cols=gamma_h
                    -wd * Bh.T @ CGe,                     # Khe  (with FEniCS minus)
                    wd * Bl.T @ CGe,                      # Kle
                    wd * Ge.T @ CGe)                      # Dee
        out = [qp(q) for q in range(Nq.shape[0])]
        return tuple(sum(o[k] for o in out) for k in range(6))
    return jax.jit(jax.vmap(kernel))


_KERNELS = {}


def _kernel(kind, dim3):
    key = (kind, dim3)
    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(kind, dim3)
    return _KERNELS[key]


# ============================================================================= assembly
def _elem_C(mat_param, mat_id, frames):
    """Per-element rotated 6x6 C: orthotropic C in the material frame rotated by the
    (beam-first) per-element [e1,e2,e3] frame -- reuses the VALIDATED solid_timo pair."""
    Cm = jnp.stack([build_orthotropic_C(jnp.asarray(mat_param[i])) for i in range(len(mat_param))])
    Ce = Cm[jnp.asarray(mat_id)]
    return jax.vmap(rotate_C_with_matrix)(Ce, jnp.asarray(frames))


def assemble_blocks(nodes, batches, mat_param, dim3):
    """Batched assembly over ALL cell-type batches -> one global system (3 dofs/node):
    scipy CSR Dhh, Dll, Dhl (N3xN3) + dense Dhe, Dle (N3,4), Dee (4,4).
    Each type is one vmapped JIT batch; COO triplets concatenate (fea-in-jax pattern)."""
    N3 = 3 * len(nodes)
    rows_h, cols_h, vals_hh, vals_ll, vals_hl = [], [], [], [], []
    Dhe = np.zeros((N3, 4))
    Dle = np.zeros((N3, 4))
    Dee = np.zeros((4, 4))
    X = jnp.asarray(nodes)
    for kind, (conn, mat_id, frames) in batches.items():
        Ce = _elem_C(mat_param, mat_id, frames)
        xe = X[jnp.asarray(conn)]                          # (n, nn, 3)
        Khh, Kll, Khl, Khe, Kle, De = _kernel(kind, dim3)(xe, Ce)
        nn = conn.shape[1]
        edof = (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), 3 * nn)
        R = np.repeat(edof[:, :, None], 3 * nn, axis=2).ravel()
        Cc = np.repeat(edof[:, None, :], 3 * nn, axis=1).ravel()
        rows_h.append(R); cols_h.append(Cc)
        vals_hh.append(np.asarray(Khh).ravel())
        vals_ll.append(np.asarray(Kll).ravel())
        vals_hl.append(np.asarray(Khl).ravel())
        np.add.at(Dhe, edof.ravel(), np.asarray(Khe).reshape(-1, 4))
        np.add.at(Dle, edof.ravel(), np.asarray(Kle).reshape(-1, 4))
        Dee += np.asarray(De).sum(0)
    R = np.concatenate(rows_h); Cc = np.concatenate(cols_h)

    def csr(vals):
        return coo_matrix((np.concatenate(vals), (R, Cc)), shape=(N3, N3)).tocsr()
    return csr(vals_hh), csr(vals_ll), csr(vals_hl), Dhe, Dle, Dee


# ==================================================================== Timoshenko finalize
def _finalize(D_eff, B_tim, C_tim):
    """FEniCS-literal Timoshenko finalize: Q_tim/G_tim/Y_tim/A_tim -> 6x6 reorder
    [EA, GA2, GA3, GJ, EI2, EI3] ordering (Deff_srt)."""
    Ainv = np.linalg.inv(D_eff)
    Q = Ainv @ np.array([(0.0, 0.0), (0.0, 0.0), (0.0, -1.0), (1.0, 0.0)])
    Ginv = Q.T @ (C_tim - B_tim.T @ Ainv @ B_tim) @ Q
    G = np.linalg.inv(Ginv)
    Y = B_tim.T @ Q @ G
    A = D_eff + Y @ Ginv @ Y.T
    S = np.zeros((6, 6))
    S[0, 0] = A[0, 0]; S[0, 1:3] = Y[0, :]; S[0, 3:6] = A[0, 1:4]
    S[3:6, 3:6] = A[1:4, 1:4]; S[3:6, 1:3] = Y[1:4, :]; S[3:6, 0] = A[1:4, 0]
    S[1:3, 1:3] = G; S[1:3, 3:6] = Y.T[:, 1:4]; S[1:3, 0] = Y.T[:, 0]
    return S


# ======================================================================== boundary solve
def solve_boundary(bnd):
    """2-D solid boundary SG (gamma_h dim=2) with the 4-mode nullspace (3 translations +
    twist) as a bordered KKT, SuperLU.  Returns (Deff6, V0 (N3,4), V1s (N3,4))."""
    nodes = bnd["nodes"]
    N3 = 3 * len(nodes)
    Dhh, Dll, Dhl, Dhe, Dle, Dee = assemble_blocks(nodes, bnd["batches"], bnd["mat_param"], dim3=False)

    Rm = np.zeros((N3, 4))                                 # 4 rigid modes
    Rm[0::3, 0] = 1.0; Rm[1::3, 1] = 1.0; Rm[2::3, 2] = 1.0
    Rm[1::3, 3] = -nodes[:, 2]; Rm[2::3, 3] = nodes[:, 1]  # twist [0, -x3, x2]
    Rm /= np.linalg.norm(Rm, axis=0, keepdims=True)
    sc = abs(Dhh.diagonal()).mean() or 1.0
    A = bmat([[Dhh, sc * csr_matrix(Rm)], [sc * csr_matrix(Rm.T), None]], format="csc")
    lu = splu(A)
    proj = lambda F: F - Rm @ (Rm.T @ F)                   # nullspace removal (like boundary_null.remove)

    V0 = np.zeros((N3, 4))
    for p in range(4):
        F = proj(Dhe[:, p])
        V0[:, p] = lu.solve(np.concatenate([F, np.zeros(4)]))[:N3]

    D1 = V0.T @ (-Dhe)
    D_eff = Dee + D1                                       # boundary: no /L

    DhlV0 = Dhl.T @ V0
    DhlTV0Dle = Dhl @ V0 + Dle
    V0DllV0 = V0.T @ (Dll @ V0)
    b = DhlTV0Dle - DhlV0
    V1s = np.zeros((N3, 4))
    for p in range(4):
        F = proj(b[:, p])
        V1s[:, p] = lu.solve(np.concatenate([F, np.zeros(4)]))[:N3]

    B_tim = DhlTV0Dle.T @ V0
    C_tim = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T)
    return _finalize(D_eff, B_tim, C_tim), V0, V1s


# ========================================================================= segment solve
def solve_segment(seg, bL, bR, V0L, V1L, V0R, V1R):
    """3-D segment SG (gamma_h dim=3) with Dirichlet BCs = the boundary fluctuations
    scattered through node2seg; one SuperLU factorization reused for the 4 V0 + 4 V1
    cases.  Returns (Deff6_segment, dof, ndof_free)."""
    nodes = seg["nodes"]
    N3 = 3 * len(nodes)
    Dhh, Dll, Dhl, Dhe, Dle, Dee = assemble_blocks(nodes, seg["batches"], seg["mat_param"], dim3=True)
    L = nodes[:, 0].max() - nodes[:, 0].min()

    def dofs_of(node_ids):
        return (3 * node_ids[:, None] + np.arange(3)[None, :]).ravel()
    bdofL = dofs_of(bL["node2seg"]); bdofR = dofs_of(bR["node2seg"])
    bdof = np.unique(np.concatenate([bdofL, bdofR]))
    free = np.setdiff1d(np.arange(N3), bdof)
    K_ff = Dhh[np.ix_(free, free)].tocsc()
    K_fb = Dhh[np.ix_(free, bdof)]
    lu = splu(K_ff)

    posL = np.searchsorted(bdof, bdofL)
    posR = np.searchsorted(bdof, bdofR)

    def bc_vec(WL, WR, p):
        ub = np.zeros(len(bdof))
        ub[posL] = WL[:, p]
        ub[posR] = WR[:, p]                                # R overwrites shared dofs (none expected)
        return ub

    def dirichlet(Frhs, WL, WR, p):
        ub = bc_vec(WL, WR, p)
        uf = lu.solve(Frhs[free] - K_fb @ ub)
        u = np.zeros(N3)
        u[free] = uf
        u[bdof] = ub
        return u

    V0 = np.zeros((N3, 4))
    for p in range(4):
        V0[:, p] = dirichlet(Dhe[:, p], V0L, V0R, p)

    D1 = V0.T @ (-Dhe)
    D_eff = (Dee + D1) / L
    D_eff = 0.5 * (D_eff + D_eff.T)

    DhlV0 = Dhl.T @ V0
    DhlTV0Dle = Dhl @ V0 + Dle
    V0DllV0 = V0.T @ (Dll @ V0)
    b = DhlTV0Dle - DhlV0
    V1s = np.zeros((N3, 4))
    for p in range(4):
        V1s[:, p] = dirichlet(b[:, p], V1L, V1R, p)

    B_tim = (DhlTV0Dle.T @ V0) / L
    C_tim = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim + C_tim.T) / L
    return _finalize(D_eff, B_tim, C_tim), N3, len(free)


# ============================================================================= top level
def compute_timo_taper_solid(yaml_path, verbose=True):
    """JAX solid taper: YAML -> (Deff_L 6x6, Deff_R 6x6, Deff_segment 6x6, info)."""
    return compute_timo_taper_solid_seg(read_solid_segment_yaml(yaml_path), verbose)


def hex_to_tets(conn8):
    """Split hex8 connectivity (n,8) into 6 tet4 each via the main-diagonal (0-6) scheme.
    In a structured grid with consistent local orderings the implied face diagonals MATCH
    between adjacent split hexes (bottom 0-2 / top 4-6 / sides 0-5, 1-6, 3-6, 0-7), so the
    tet region is conforming; the hex|tet interface (quad face vs 2 tris on the same 4
    nodes) is the standard node-tied hex-dominant transition."""
    c = np.asarray(conn8)
    T = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    return np.concatenate([c[:, list(t)] for t in T], axis=0)


def split_batches_to_tets(seg, mask=None):
    """New seg dict with the masked hex8 elements (default: all) split into tet4 (6 each,
    inheriting material + frame).  mask: bool (n_hex,) selecting hexes to split."""
    out = dict(seg)
    batches = dict(seg["batches"])
    conn, mat, frm = batches["hex8"]
    m = np.ones(len(conn), bool) if mask is None else np.asarray(mask, bool)
    new_tets = hex_to_tets(conn[m])
    tet_mat = np.tile(mat[m], 6)
    tet_frm = np.tile(frm[m], (6, 1))
    if "tet4" in batches:
        t0, m0, f0 = batches["tet4"]
        new_tets = np.concatenate([t0, new_tets]); tet_mat = np.concatenate([m0, tet_mat])
        tet_frm = np.concatenate([f0, tet_frm])
    batches["tet4"] = (new_tets, tet_mat, tet_frm)
    if m.all():
        batches.pop("hex8")
    else:
        batches["hex8"] = (conn[~m], mat[~m], frm[~m])
    out["batches"] = batches
    out["nelem"] = sum(len(b[0]) for b in batches.values())
    return out


def compute_timo_taper_solid_seg(seg, verbose=True):
    """As compute_timo_taper_solid but from an in-memory seg dict (read_solid_segment_yaml
    format) -- used for hybrid hex+tet variants built by split_batches_to_tets."""
    t0 = time.time()
    bL = extract_boundary_submesh(seg, "L")
    bR = extract_boundary_submesh(seg, "R")
    t_read = time.time() - t0
    if verbose:
        bt = {k: v[0].shape for k, v in seg["batches"].items()}
        b2 = {k: v[0].shape for k, v in bL["batches"].items()}
        print("segment batches: %s ; L-boundary: %s (%d nodes)" % (bt, b2, len(bL["nodes"])), flush=True)

    t0 = time.time()
    DL, V0L, V1L = solve_boundary(bL)
    DR, V0R, V1R = solve_boundary(bR)
    t_bnd = time.time() - t0

    t0 = time.time()
    Dseg, dof, nfree = solve_segment(seg, bL, bR, V0L, V1L, V0R, V1R)
    t_seg = time.time() - t0
    info = dict(dof=dof, nfree=nfree, t_read=t_read, t_boundary=t_bnd, t_segment=t_seg,
                nelem=seg["nelem"])
    if verbose:
        print("JAX solid taper: dof=%d (free %d)  boundary %.2fs  segment %.2fs" %
              (dof, nfree, t_bnd, t_seg), flush=True)
    return DL, DR, Dseg, info
