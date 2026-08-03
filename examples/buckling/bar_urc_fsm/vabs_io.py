'''vabs_io.py -- reader for VABS ".glb" global-response (dehomogenization input) files.

The .glb layout is NOT guessed: it is fixed by our own writers, which round-trip against the
BAR-URC reference files.  Cross-checked against three independent artifacts in this repo:

  * examples/data/iea_all_stations/dehom_iea/gen_glb_from_beamdyn.py :: write_glb()
  * examples/TW-paper/xsec_paper/make_span_glb.py                    (inline writer)
  * examples/TW-paper/xsec_paper/read_loads_grid.py :: read_glb()    (inline reader)

------------------------------------------------------------------------------------------------
EXACT LINE-BY-LINE LAYOUT (10 non-blank lines):

  line 1  (idx 0) :  u1 u2 u3                 beam reference-point displacement, VABS frame  [m]
  line 2  (idx 1) :  C11 C12 C13     |
  line 3  (idx 2) :  C21 C22 C23     |--- 3x3 direction-cosine matrix of the deformed section triad
  line 4  (idx 3) :  C31 C32 C33     |    (C = B C_bd B^T from the BeamDyn Wiener-Milenkovic rot)
  line 5  (idx 4) :  F1  M1  M2  M3          <-- 4 values: axial force + torsion + 2 bending moments
  line 6  (idx 5) :  F2  F3                  <-- 2 values: the two TRANSVERSE SHEAR forces
  lines 7-10      :  0 0 0 0 0 0   (x4)      sectional-load DERIVATIVES d^n(F,M)/dx1^n, n=1..4
                                             (all zero for static / non-tapered recovery)

The six load components therefore WRAP 4-then-2 across lines 5 and 6, and the wrap is NOT
[F1 F2 F3 M1] + [M2 M3].  It is [F1 M1 M2 M3] + [F2 F3].

FRAME / SIGN CONVENTION (VABS beam frame):
  x1 = beam (spanwise) axis;  x2, x3 = the two cross-section in-plane axes.
  F1 = axial force        M1 = torsional moment  (about the beam axis)
  F2 = transverse shear   M2 = bending moment about x2   (flapwise for a blade)
  F3 = transverse shear   M3 = bending moment about x3   (edgewise for a blade)
Resultants are in the section-LOCAL frame; the line-2..4 DCM rotates the recovered field back
to global.  From BeamDyn: [F1,F2,F3,M1,M2,M3] = [FzL, -FyL, FxL, MzL, -MyL, MxL].
------------------------------------------------------------------------------------------------
'''
try:                                  # numpy is optional: the parse is pure-python, arrays are a
    import numpy as np                # convenience for the downstream dehom code.
except ImportError:                   # pragma: no cover
    np = None


def _arr(x):
    return np.array(x, float) if np is not None else x


def read_glb(path):
    '''Read a VABS .glb file.

    Returns dict with
        u  : (3,)   reference-point displacement
        C  : (3,3)  direction-cosine matrix
        F  : (3,)   [F1, F2, F3]  = [axial, shear2, shear3]
        M  : (3,)   [M1, M2, M3]  = [torsion, bend2, bend3]
        FF : (6,)   [F1,F2,F3,M1,M2,M3]  (VABS order, as used by our dehom code)
        dload : (n,6) trailing sectional-load-derivative rows (zeros for static recovery)
    '''
    rows = [ln.split() for ln in open(path) if ln.strip()]
    u = [float(x) for x in rows[0]]
    C = [[float(x) for x in rows[r]] for r in (1, 2, 3)]
    F1, M1, M2, M3 = (float(x) for x in rows[4])      # line 5: 4 values
    F2, F3 = (float(x) for x in rows[5])              # line 6: 2 values
    F = [F1, F2, F3]
    M = [M1, M2, M3]
    dload = [[float(x) for x in r] for r in rows[6:]]
    return dict(u=_arr(u), C=_arr(C), F=_arr(F), M=_arr(M),
                FF=_arr(F + M), dload=_arr(dload))


def write_glb(path, u, C, F, M, nderiv=4):
    '''Inverse of read_glb; matches gen_glb_from_beamdyn.write_glb byte-for-byte in layout.'''
    with open(path, 'w') as f:
        f.write('%.10g %.10g %.10g\n' % tuple(u))
        for r in range(3):
            f.write('%.16g %.16g %.16g\n' % tuple(C[r]))
        f.write('%.10g %.10g %.10g %.10g\n' % (F[0], M[0], M[1], M[2]))
        f.write('%.10g %.10g\n' % (F[1], F[2]))
        for _ in range(nderiv):
            f.write('0 0 0 0 0 0\n')


# ================================================================================================
# VABS INPUT (.in) + DEHOMOGENIZATION OUTPUT (.SM/.S/.E/.EM + *N variants, .U, .ELE, .K) READERS
# ================================================================================================
# Established by direct inspection of the BAR-URC station-15 run
#   .../bar_urc_flap_1200_vabs_npl_2_ar_2p5/temp/bar_urc-15-t-0.in[.EXT]
# whose header declares  nnode = 8158,  nelem = 7513 (all 4-node quads),  nmate = 6,  nlayer = 162.
# Every row count below was verified against those three numbers:
#
#   file                rows      = ?                what a row is
#   ------------------  --------  -----------------  --------------------------------------------
#   .U                  8158      = nnode            one NODE
#   .ELE                7513      = nelem            one ELEMENT (element-averaged)
#   .SM .S .EM .E       30052     = 4 x nelem        one GAUSS POINT (2x2 rule on a quad)
#   .SMN .SN .EMN .EN   30052     = sum(valence)     one ELEMENT-NODE pair (discontinuous)
#
# The 2x2-Gauss claim is not an inference from the count alone.  Element 1 has corner nodes
# 1,3,31,30 spanning y2 in [-2.08218, -2.06828]; the first four .SM rows sit at
# y2 = -2.07924, -2.07923, -2.07123, -2.07122 -- i.e. exactly the two +/-1/sqrt(3) stations in y2,
# each appearing twice.  A triangular mesh gives 3 x nelem instead.
#
# COMPONENT ORDER (columns after the coordinates) is VABS upper-triangular row-major
#       11, 12, 13, 22, 23, 33
# NOT the [11,22,33,23,13,12] Voigt order used inside our dehom code.  Use VOIGT_FROM_VABS
# (or pass order='voigt') to convert.  This matches the reorder already hard-coded in
# examples/TW-paper/xsec_paper/dehom_st15_figs.py::load_sm  ->  d[:, 2:8][:, [0,3,5,4,2,1]].
#
# HEADER LINES: this VABS build writes the .SM/.S/.E/.EM families with NO header, but other
# builds (see xsec_paper/dehom_r020_figs.py, which uses skiprows=2 on iea_r020.sg.SM) emit two
# title lines.  The readers below auto-detect and skip any leading non-numeric lines, so the
# same call works for both.
# ------------------------------------------------------------------------------------------------

#: VABS writes the 6 tensor components as 11, 12, 13, 22, 23, 33 (upper triangle, row-major).
VABS_COMP_ORDER = ('11', '12', '13', '22', '23', '33')

#: Index array mapping VABS order -> our internal Voigt order [11, 22, 33, 23, 13, 12].
VOIGT_FROM_VABS = [0, 3, 5, 4, 2, 1]

#: Number of 5-line-ish scalars per material record, keyed by the VABS orthotropy flag.
#: 0 = isotropic (E, nu, rho); 1 = orthotropic (E1..3, G12/13/23, nu12/13/23, rho);
#: 2 = general anisotropic (21 independent C_ij, rho).
_MAT_NVAL = {0: 3, 1: 10, 2: 22}


def _need_numpy():
    if np is None:                                                        # pragma: no cover
        raise ImportError('numpy is required for the .in/.SM/.U/.K readers')


def _is_float(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _skiprows_auto(path):
    '''Number of leading lines that are NOT pure numeric rows (VABS title/header lines).'''
    n = 0
    with open(path) as f:
        for ln in f:
            tok = ln.split()
            if tok and all(_is_float(t) for t in tok):
                return n
            n += 1
    return n


def _load_auto(path):
    _need_numpy()
    return np.atleast_2d(np.loadtxt(path, skiprows=_skiprows_auto(path)))


def read_sm(path, order='vabs'):
    '''Read any VABS point-wise stress/strain recovery file.

    Handles BOTH families with one call, distinguished by column count:

      * 8 columns  -- .SM (material-frame stress), .S (global-frame stress),
                      .EM (material-frame strain), .E (global-frame strain).
                      Layout:  y2  y3  c11 c12 c13 c22 c23 c33
                      One row per GAUSS POINT, in element order (4 per quad, 3 per triangle).

      * 9 columns  -- .SMN / .SN / .EMN / .EN, the nodal ("N") variants.
                      Layout:  node_id  y2  y3  c11 c12 c13 c22 c23 c33
                      One row per ELEMENT-NODE pair, sorted ascending by node_id.  A node
                      therefore appears once per adjacent element (its valence) and the values
                      are NOT averaged -- they are the discontinuous per-element extrapolations.
                      Verified on station 15: 30052 rows, 8158 unique ids, min 1, max 8158.

    Parameters
    ----------
    path  : file to read.
    order : 'vabs'  -> columns left as written, [11, 12, 13, 22, 23, 33];
            'voigt' -> reordered to our internal [11, 22, 33, 23, 13, 12].

    Returns
    -------
    dict with
        xy    : (n, 2)  in-plane coordinates (y2, y3) of each point
        comp  : (n, 6)  the six tensor components in the requested order
        node  : (n,) int node ids for the 9-column nodal variants, else None
        nodal : bool, True for the ".*N" element-node files
        order : the order string actually applied
    '''
    d = _load_auto(path)
    if d.shape[1] == 9:
        node, xy, comp, nodal = d[:, 0].astype(int), d[:, 1:3], d[:, 3:9], True
    elif d.shape[1] == 8:
        node, xy, comp, nodal = None, d[:, 0:2], d[:, 2:8], False
    else:
        raise ValueError('%s: expected 8 (gauss) or 9 (nodal) columns, got %d'
                         % (path, d.shape[1]))
    if order == 'voigt':
        comp = comp[:, VOIGT_FROM_VABS]
    elif order != 'vabs':
        raise ValueError("order must be 'vabs' or 'voigt', got %r" % (order,))
    return dict(xy=xy, comp=comp, node=node, nodal=nodal, order=order)


def read_u(path):
    '''Read a VABS .U dehomogenized-displacement file.

    Layout: 6 columns, ONE ROW PER NODE, node ids running 1..nnode in order (verified: 8158
    rows, ids strictly 1..8158 for station 15):

        node_id   y2   y3   U1   U2   U3

    (y2, y3) repeat the undeformed cross-section coordinates from the .in node block, so a .U
    file is self-sufficient as a point cloud.  U1/U2/U3 are the 3-D displacements of that
    material point in the VABS beam frame (x1 = beam axis), i.e. the beam displacement plus the
    warping field -- the same quantity our RM dehom must reproduce.

    Returns dict with  node (n,) int,  xy (n, 2),  u (n, 3).
    '''
    d = _load_auto(path)
    if d.shape[1] != 6:
        raise ValueError('%s: expected 6 columns (node y2 y3 U1 U2 U3), got %d'
                         % (path, d.shape[1]))
    return dict(node=d[:, 0].astype(int), xy=d[:, 1:3], u=d[:, 3:6])


def read_ele(path, order='vabs'):
    '''Read a VABS .ELE element-averaged recovery file.

    Layout: 25 columns, ONE ROW PER ELEMENT (verified: 7513 rows = nelem):

        elem_id | strain_global(6) | stress_global(6) | strain_material(6) | stress_material(6)

    so it bundles what .E, .S, .EM and .SM carry separately, but averaged over the element
    instead of sampled at Gauss points.  Components use the same 11,12,13,22,23,33 order.

    Returns dict with elem (n,) int and strain_g / stress_g / strain_m / stress_m, each (n, 6).
    '''
    d = _load_auto(path)
    if d.shape[1] != 25:
        raise ValueError('%s: expected 25 columns, got %d' % (path, d.shape[1]))
    blocks = [d[:, 1:7], d[:, 7:13], d[:, 13:19], d[:, 19:25]]
    if order == 'voigt':
        blocks = [b[:, VOIGT_FROM_VABS] for b in blocks]
    elif order != 'vabs':
        raise ValueError("order must be 'vabs' or 'voigt', got %r" % (order,))
    return dict(elem=d[:, 0].astype(int), strain_g=blocks[0], stress_g=blocks[1],
                strain_m=blocks[2], stress_m=blocks[3], order=order)


def read_vabs_in(path):
    '''Read a VABS input (.in) file -- the 2-D solid mesh, layup and material database.

    Parsed as a TOKEN STREAM (whitespace-delimited) rather than line-by-line, because the
    material records wrap differently between VABS writers.  Structure, in order:

        format_flag  nlayer                                   e.g. "1 162"
        Timoshenko_flag  recover_flag  thermal_flag           e.g. "1 0 0"
        curve_flag  oblique_flag  trapeze_flag  Vlasov_flag   e.g. "0 0 0 0"
        [k11 k12 k13]                                         only if curve_flag  == 1
        [cos1 cos2]                                           only if oblique_flag == 1
        nnode  nelem  nmate                                   e.g. "8158 7513 6"

        nnode  x  (node_id, y2, y3)
        nelem  x  (elem_id, n1..n9)      ZERO-PADDED to 9 slots; a 4-node quad is
                                         "1 1  3  31  30 0  0  0  0  0"
        nelem  x  (elem_id, layer_id, theta3)   theta3 = in-plane ply rotation [deg]
        nlayer x  (layer_id, mat_id, theta1)    theta1 = fiber angle in the ply [deg]
        nmate  x  (mat_id, orth_flag, <constants>, rho)

    The two-level layer indirection is the important part for us: an element does NOT name a
    material directly.  It names a LAYER, and the layer table maps layer -> (material, theta1).
    Station 15 has 162 layers over 6 materials; the resolved per-element material/angles are
    returned pre-joined as elem_mat / elem_theta1 / elem_theta3 so callers never redo the join.

    An orthotropic material record (orth_flag == 1) reads, e.g. material 1 of station 15:

        1 1
        28211400000.0 16238800000.0 15835500000.0      E1  E2  E3
        8248220000.0 3491240000.0 3491240000.0         G12 G13 G23
        0.497511 0.18091 0.27481                       nu12 nu13 nu23
        1940.0                                         rho

    Returns
    -------
    dict with
        nnode, nelem, nmate, nlayer          : int counts from the header
        flags                                : dict of the header flags
        nodes      : (nnode, 2)  y2, y3 coordinates, row i = node i+1
        node_id    : (nnode,) int as written
        conn       : (nelem, 9) int, 1-based node ids, 0 = unused slot
        nnpe       : (nelem,) int nonzero slots per element -> 3 tri / 4 quad / 6 / 8 / 9
        elem_id    : (nelem,) int as written
        elem_layer : (nelem,) int layer id of each element
        elem_theta3: (nelem,) float in-plane ply rotation [deg]
        elem_mat   : (nelem,) int material id, resolved through the layer table
        elem_theta1: (nelem,) float fiber angle [deg], resolved through the layer table
        layers     : dict layer_id -> (mat_id, theta1)
        materials  : dict mat_id -> dict(orth=..., rho=..., plus E/G/nu or C)
        contour    : list of (y2, y3) of the nodes on the free boundary, ordered into loops
                     (see boundary_loops); the OUTER loop is loops[0]
    '''
    _need_numpy()
    with open(path) as f:
        tok = f.read().split()
    i = 0

    def take(n, cast=float):
        nonlocal i
        out = [cast(t) for t in tok[i:i + n]]
        if len(out) != n:
            raise ValueError('%s: truncated at token %d' % (path, i))
        i += n
        return out

    format_flag, nlayer = take(2, int)
    timoshenko_flag, recover_flag, thermal_flag = take(3, int)
    curve_flag, oblique_flag, trapeze_flag, vlasov_flag = take(4, int)
    curvature = take(3) if curve_flag == 1 else [0.0, 0.0, 0.0]
    oblique = take(2) if oblique_flag == 1 else None
    if thermal_flag:                                                      # pragma: no cover
        raise NotImplementedError('%s: thermal_flag=%d adds CTE records this reader does not '
                                  'parse yet' % (path, thermal_flag))
    nnode, nelem, nmate = take(3, int)

    nod = np.array(take(3 * nnode)).reshape(nnode, 3)
    node_id, nodes = nod[:, 0].astype(int), nod[:, 1:3]

    ele = np.array(take(10 * nelem, int)).reshape(nelem, 10)
    elem_id, conn = ele[:, 0], ele[:, 1:10]
    nnpe = (conn != 0).sum(axis=1)

    lay = np.array(take(3 * nelem)).reshape(nelem, 3)
    elem_layer, elem_theta3 = lay[:, 1].astype(int), lay[:, 2]

    if nlayer > 0:
        lt = np.array(take(3 * nlayer)).reshape(nlayer, 3)
        layers = {int(r[0]): (int(r[1]), float(r[2])) for r in lt}
        lmat = np.zeros(nlayer + 1, int)
        lth1 = np.zeros(nlayer + 1)
        for k, (m, t) in layers.items():
            lmat[k], lth1[k] = m, t
        elem_mat, elem_theta1 = lmat[elem_layer], lth1[elem_layer]
    else:                                                                 # format_flag 0 style
        layers = {}
        elem_mat, elem_theta1 = elem_layer, np.zeros(nelem)

    materials = {}
    for _ in range(nmate):
        mid, orth = take(2, int)
        if orth not in _MAT_NVAL:                                         # pragma: no cover
            raise ValueError('%s: material %d has unknown orthotropy flag %d' % (path, mid, orth))
        v = take(_MAT_NVAL[orth])
        if orth == 0:
            materials[mid] = dict(orth=0, E=v[0], nu=v[1], rho=v[2])
        elif orth == 1:
            materials[mid] = dict(orth=1, E=v[0:3], G=v[3:6], nu=v[6:9], rho=v[9])
        else:
            materials[mid] = dict(orth=2, C=v[0:21], rho=v[21])

    out = dict(format_flag=format_flag, nlayer=nlayer, nnode=nnode, nelem=nelem, nmate=nmate,
               flags=dict(timoshenko=timoshenko_flag, recover=recover_flag,
                          thermal=thermal_flag, curve=curve_flag, oblique=oblique_flag,
                          trapeze=trapeze_flag, vlasov=vlasov_flag),
               curvature=_arr(curvature), oblique=oblique,
               node_id=node_id, nodes=nodes, elem_id=elem_id, conn=conn, nnpe=nnpe,
               elem_layer=elem_layer, elem_theta3=elem_theta3,
               elem_mat=elem_mat, elem_theta1=elem_theta1,
               layers=layers, materials=materials)
    out['loops'] = boundary_loops(out)
    out['contour'] = out['loops'][0] if out['loops'] else None
    return out


def boundary_loops(mesh):
    '''Extract the cross-section CONTOUR from a read_vabs_in() result.

    An edge shared by exactly one element is on the boundary; chaining those edges gives the
    outer profile plus one loop per internal cavity (for a webbed blade section: the outer
    airfoil skin plus each enclosed cell).  Loops are returned longest-first, so loops[0] is the
    outer contour.  Each loop is an (m, 2) array of y2/y3 coordinates.
    '''
    _need_numpy()
    conn, nnpe, nodes = mesh['conn'], mesh['nnpe'], mesh['nodes']
    corners = {3: 3, 4: 4, 6: 3, 8: 4, 9: 4}                # mid-side nodes trail the corners
    count, order = {}, {}
    for e in range(len(conn)):
        nc = corners.get(int(nnpe[e]))
        if nc is None:                                                    # pragma: no cover
            continue
        cs = conn[e, :nc]
        for a in range(nc):
            u, v = int(cs[a]), int(cs[(a + 1) % nc])
            key = (u, v) if u < v else (v, u)
            count[key] = count.get(key, 0) + 1
            order.setdefault(key, (u, v))
    edges = [order[k] for k, c in count.items() if c == 1]

    nxt = {}
    for u, v in edges:
        nxt.setdefault(u, []).append(v)
        nxt.setdefault(v, []).append(u)
    seen, loops = set(), []
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur, prev = [start], start, None
        seen.add(start)
        while True:
            cand = [w for w in nxt.get(cur, []) if w != prev and w not in seen]
            if not cand:
                break
            prev, cur = cur, cand[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) > 2:
            loops.append(np.array([nodes[n - 1] for n in loop]))
    loops.sort(key=len, reverse=True)
    return loops


# ------------------------------------------------------------------------------------------------
# .K -- the homogenization result (mass, classical 4x4, Timoshenko 6x6, centers)
# ------------------------------------------------------------------------------------------------
# Free-form report, organised as
#       <title line>
#       ======================================================
#       <blank>
#       <numeric rows ...>
# with loose "Name = value" scalars interleaved.  read_k() walks the '=====' rulers to attach
# each numeric block to the title above it, and separately harvests every "... = <float>" line.
#
# *** THE GJ GOTCHA IS REAL AND VISIBLE IN THIS FILE. ***
# "The torsional stiffness GJ" is printed from the CLASSICAL (Euler-Bernoulli 4x4) solution and
# is NOT the Timoshenko K[3,3].  For station 15:
#       classical GJ  = 1.3157769625E+08     (classical4[1,1], and the scalar 'GJ')
#       Timoshenko K44= 1.5604378583E+08     (timo6[3,3])
# an 18.6% difference.  Compare our RM twist stiffness against timo6[3,3], never against 'GJ'.
#
# Timoshenko 6x6 DOF order is  (1 extension, 2-3 shear, 4 twist, 5-6 bending)  -- so the twist
# entry sits at index 3, BETWEEN the two shears and the two bendings, which is NOT the
# [F1,F2,F3,M1,M2,M3] ordering used by the .glb loads above.  They coincide only because
# M1 == twist is also index 3 there; the classical 4x4 order differs again:
# (1 extension, 2 twist, 3-4 bending).
# ------------------------------------------------------------------------------------------------

_K_SECTIONS = (
    ('The 6X6 Mass Matrix at the Mass Center', 'mass6_center'),      # test before the shorter one
    ('The 6X6 Mass Matrix', 'mass6'),
    ('The Mass Center of the Cross Section', 'mass_center'),
    ('The Geometric Center of the Cross Section', 'geo_center'),
    ('Classical Stiffness Matrix', 'classical4'),
    ('Classical Compliance Matrix', 'classical_compliance4'),
    ('The Tension Center of the Cross Section', 'tension_center'),
    ('Timoshenko Stiffness Matrix', 'timo6'),
    ('Timoshenko Compliance Matrix', 'timo_compliance6'),
    ('The Shear Center of the Cross Section', 'shear_center'),
)


def read_k(path):
    '''Read a VABS .K homogenization report.

    Returns a dict whose headline entries are

        timo6                 (6, 6)  Timoshenko stiffness, order (ext, shear2, shear3, twist,
                                      bend5, bend6)  -- twist is index 3
        timo_compliance6      (6, 6)
        classical4            (4, 4)  Euler-Bernoulli, order (ext, twist, bend3, bend4)
        classical_compliance4 (4, 4)
        mass6, mass6_center   (6, 6)  mass matrix at the origin / at the mass center
        mass_center, geo_center, tension_center, shear_center   (2,) each
        scalars               dict of every "Name = value" line (Area, EA, GJ, EI22, EI33,
                              GA22, GA33, mass per unit span, radii of gyration, ...)
        angles                dict of the "rotated ... by X degrees" values [deg]

    plus convenience aliases  area, EA, GJ_classical, GJ_timo, EI22, EI33, GA22, GA33.
    Note GJ_classical != GJ_timo (see the module comment above): GJ_timo is timo6[3, 3].
    '''
    _need_numpy()
    with open(path) as f:
        lines = f.read().splitlines()

    out, scalars, angles = {}, {}, {}
    for j, ln in enumerate(lines):
        s = ln.strip()

        if s.startswith('===') and j > 0:                       # ruler -> title is the line above
            title = lines[j - 1].strip()
            key = None
            for pat, name in _K_SECTIONS:
                if title.startswith(pat):
                    key = name
                    break
            if key is None or key in out:
                continue
            rows = []
            for ln2 in lines[j + 1:]:
                t = ln2.split()
                if t and all(_is_float(x) for x in t):
                    rows.append([float(x) for x in t])
                elif rows:                                      # block ended
                    break
            if rows:
                a = np.array(rows)
                out[key] = a.ravel() if a.shape[0] == 1 else a

        # Scalars appear in TWO styles and both must be caught:
        #     "Area =    4.0346011271E-01"                      (with '=')
        #     "The extension stiffness EA    1.3082688863E+10"   (NO '=' -- EA/GJ/EI22/EI33)
        # Rule: last token is a float, first token is not (that excludes matrix rows), and the
        # line is not a ruler.  Trailing '=' is stripped from the harvested label.
        t = s.split()
        if len(t) >= 2 and not s.startswith('===') and _is_float(t[-1]) and not _is_float(t[0]):
            label = ' '.join(t[:-1]).rstrip('=').strip()
            if label:
                scalars[label] = float(t[-1])

        if 'rotated' in s and 'degrees' in s:
            for t in s.split():
                if _is_float(t):
                    angles['principal_inertial' if 'inertial' in s else
                           ('principal_bending' if 'bending' in s else 'principal_shear')] = float(t)
                    break

    def pick(*names):
        for n in names:
            for k, v in scalars.items():
                if k.startswith(n) or k.endswith(n) or n in k:
                    return v
        return None

    out['scalars'] = scalars
    out['angles'] = angles
    out['area'] = pick('Area')
    out['EA'] = pick('The extension stiffness EA', 'extension stiffness')
    out['GJ_classical'] = pick('The torsional stiffness GJ', 'torsional stiffness')
    out['GJ_timo'] = float(out['timo6'][3, 3]) if 'timo6' in out else None
    out['EI22'] = pick('Principal bending stiffness EI22')
    out['EI33'] = pick('Principal bending stiffness EI33')
    out['GA22'] = pick('Principal shear stiffness GA22')
    out['GA33'] = pick('Principal shear stiffness GA33')
    out['mass_per_span'] = pick('Mass per unit span')
    return out


if __name__ == '__main__':
    import os
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else (
        r'C:\Users\bagla0\OneDrive - purdue.edu\2025_195\20250314_RunallSolidsegments'
        r'\bar_urc_vabs_runs\bar_urc_flap_1200_vabs_npl_2_ar_2p5\temp')
    print('BAR-URC .glb sectional resultants  (VABS frame, x1 = beam axis)')
    print('dir: %s\n' % d)
    hdr = ('%-4s %12s %12s %12s %14s %14s %14s'
           % ('st', 'F1 axial', 'F2 shear', 'F3 shear', 'M1 torsion', 'M2 bend', 'M3 bend'))
    print(hdr)
    print('%-4s %12s %12s %12s %14s %14s %14s' % ('', '[N]', '[N]', '[N]', '[N-m]', '[N-m]', '[N-m]'))
    print('-' * len(hdr))
    for i in range(30):
        p = os.path.join(d, 'bar_urc-%d-t-0.in.glb' % i)
        if not os.path.exists(p):
            print('%-4d  (missing)' % i)
            continue
        g = read_glb(p)
        print('%-4d %12.4e %12.4e %12.4e %14.5e %14.5e %14.5e'
              % (i, g['F'][0], g['F'][1], g['F'][2], g['M'][0], g['M'][1], g['M'][2]))
