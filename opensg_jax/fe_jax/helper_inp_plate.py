"""helper_inp_plate.py -- Abaqus .inp GENERATORS for the cylindrical-bending plate
strip: the MSG-RM 8x8 section, and the community-standard FSDT composite section.

Both write the same, complete, submit-ready model (mesh, sinusoidal load, cylindrical-
bending BCs, FF-extraction output requests) and differ ONLY in the shell section:

  write_plate_strip_inp        *SHELL GENERAL SECTION carrying the MSG-RM 8x8
                               ([[A,B],[B,D]] 21 constants, Abaqus triangular order)
                               + *TRANSVERSE SHEAR STIFFNESS with the MSG 2x2 G.
                               FSDT KINEMATICS (S4 is a first-order shear-deformable
                               element) with MSG SECTION PHYSICS.
  write_plate_strip_inp_fsdt   the GENERAL-COMMUNITY FSDT route: *SHELL SECTION,
                               COMPOSITE with the ply stack and *ELASTIC, TYPE=LAMINA
                               materials.  Abaqus builds A/B/D by classical lamination
                               and its OWN transverse-shear stiffness estimate from
                               the layup -- exactly what a practitioner gets by
                               defining plies in any commercial FE shell model.

mesh       0 <= x <= a, one square S4 element wide, ``nel`` elements along the span,
           reference surface = the laminate mid-surface
load       q(x) = q0 sin(pi x / a), piecewise-constant per element (centre value)
BCs        cylindrical bending: u2 = ur1 = ur3 = 0 on ALL nodes; u3 = 0 on both end
           edges, u1 = 0 at x = 0
output     field SF/SM/U to the .odb; .dat prints of U at the mid-span node (NMID)
           and SF/SM in the mid-span (EMID) and end (EEND) elements -- the two
           FF-extraction stations (M max at a/2, Q max at 0)

VALIDATED (MSG route) against a real Abaqus 2024 run (Garg caseA, S = 10): mid-span
U3 94.3695e-6 m vs the closed form q0/(p^4 D11) + q0/(p^2 G11) = 94.3817e-6 (0.013%);
IP resultants M11 = 1012.92 vs q0/p^2 = 1013.21, Q1 = 3182.44 vs q0/p = 3183.10.

GOTCHA (cost a failed submit): Abaqus rejects an input PATH containing whitespace
("Command line option 'input' value must not contain whitespace") -- e.g. anything
under "OneDrive - purdue.edu".  Run the job from a space-free working directory and
copy results back.

Report extraction map (Abaqus .rpt column names -> the example-7 FF vector):
    FF = [SF1, SF2, SF3, SM1, SM2, SM3, SF4, SF5]
       = [N11, N22, N12, M11, M22, M12, Q1,  Q2]
(the .rpt prints columns in the order SF1 SF2 SF6 SF3 SF4 SF5 | SM2 SM1 SM3 -- read
the HEADER, not the position).  U/UR are nodal fields; SF/SM live at integration points.
"""
import numpy as np


def _fmt_data_lines(vals):
    """Abaqus data lines: at most 8 entries per line."""
    out = []
    for i in range(0, len(vals), 8):
        out.append(", ".join("%.6e" % v for v in vals[i:i + 8]))
    return out


def _strip_inp(path, title, header_lines, material_lines, section_lines, a, q0, nel,
               coupled=False):
    """The common strip model; the caller supplies the material + section blocks.

    Variables
    ---------
    path            output .inp path
    title           the *HEADING line
    header_lines    ** comment lines recorded at the top of the deck
    material_lines  the *MATERIAL block (empty for a general section)
    section_lines   the shell-section block (general 8x8 or composite stack)
    a, q0, nel      span [m], load amplitude [Pa], S4 elements along the span
    coupled         BC mode.  False (default, the orthotropic cylindrical-bending
                    set): u2 = ur1 = ur3 = 0 on ALL nodes -- exact when the layup
                    has no extension/shear or bending/twist coupling.  True (the
                    ANGLE-PLY set, e.g. the Yu-2003 laminates): shear coupling
                    makes v(x) and the twist rotation NONZERO under cylindrical
                    bending, so u2/ur1 must be left FREE and the two width rows
                    are tied node-by-node with *EQUATION on all six dofs --
                    enforcing d/dy = 0, the infinite plate.  Free y-edges are NOT
                    enough: the one-element strip then sheds the coupling twist
                    like a narrow beam in torsion (M12/Q2 average to ~0 across
                    the width; observed on the Yu-2003 case1 deck).  BCs touch
                    only the master j = 0 row (the j = 1 dofs are the eliminated
                    equation terms): u2 pinned at one node, u3 = 0 at the two
                    end nodes, u1 = 0 at x = 0, and NO drilling BC -- the
                    angle-ply ur3 ~ v,1 is nonzero and constraining it injects a
                    spurious constant N12 (S4's internal drilling stabilization
                    suffices).
    p, b            wavenumber pi/a; element width (one square element)
    nid(i, j)       node numbering: span index i (0..nel), width row j (0, 1)
    L, A            the accumulated deck lines and the append shorthand
    xc              per-element centroid x for the piecewise-constant sine load
    """
    p = np.pi / a
    b = a / nel                                   # one square element across the width
    L = []
    A = L.append
    A("*HEADING")
    A(title)
    for ln in header_lines:
        A("** " + ln)
    A("** span a = %g m, q0 = %g Pa, %d S4 elements; load q0*sin(pi*x/a) per element"
      % (a, q0, nel))
    A("*NODE")
    nid = lambda i, j: i * 2 + j + 1
    for i in range(nel + 1):
        for j in (0, 1):
            A("%d, %.8f, %.8f, 0.0" % (nid(i, j), i * a / nel, j * b))
    A("*ELEMENT, TYPE=S4, ELSET=EALL")
    for i in range(nel):
        A("%d, %d, %d, %d, %d" % (i + 1, nid(i, 0), nid(i + 1, 0),
                                  nid(i + 1, 1), nid(i, 1)))
    A("*ELSET, ELSET=EMID")
    A("%d" % (nel // 2))
    A("*ELSET, ELSET=EEND")
    A("1")
    if coupled:
        # BCs may only touch the j = 0 (master) row: the j = 1 row is the
        # ELIMINATED first term of the width-tie equations below
        A("*NSET, NSET=NX0")
        A("%d" % nid(0, 0))
        A("*NSET, NSET=NXA")
        A("%d" % nid(nel, 0))
    else:
        A("*NSET, NSET=NX0")
        A("%d, %d" % (nid(0, 0), nid(0, 1)))
        A("*NSET, NSET=NXA")
        A("%d, %d" % (nid(nel, 0), nid(nel, 1)))
    A("*NSET, NSET=NMID")
    A("%d" % nid(nel // 2, 0))
    A("*NSET, NSET=NROW0, GENERATE")
    A("%d, %d, 2" % (nid(0, 0), nid(nel, 0)))
    A("*NSET, NSET=NALL, GENERATE")
    A("1, %d, 1" % nid(nel, 1))
    A("**")
    L.extend(material_lines)
    L.extend(section_lines)
    A("**")
    if coupled:
        # width ties u(j=1) = u(j=0) on ALL SIX dofs -> d/dy = 0 exactly (the
        # infinite plate).  Without them the one-element strip has FREE y-edges
        # and sheds the coupling twist like a narrow beam in torsion (M12/Q2
        # average to ~0 across the width) -- seen live on the Yu case1 deck.
        A("** ---- width ties: infinite-plate condition for the coupled layup ----")
        A("*EQUATION")
        for i in range(nel + 1):
            for d in (1, 2, 3, 4, 5, 6):
                A("2")
                A("%d, %d, 1.0, %d, %d, -1.0" % (nid(i, 1), d, nid(i, 0), d))
        A("**")
    A("** ---- cylindrical bending + simple supports ----")
    A("*BOUNDARY")
    if coupled:
        A("%d, 2, 2" % nid(0, 0))                 # pin the u2 rigid mode only
        # NO drilling BC here: the angle-ply solution has ur3 ~ v,1 != 0, and
        # constraining it injects a spurious CONSTANT N12 (drilling reaction ->
        # membrane shear; observed as N12 = -0.018 on the Yu case1 strip while
        # the infinite plate has N12 = 0).  S4's internal drilling stabilization
        # keeps the model well-posed without it.
    else:
        A("NALL, 2, 2")                           # u2 = 0   (nothing varies with x2)
        A("NALL, 4, 4")                           # ur1 = 0  (kappa22 = 0)
        A("NALL, 6, 6")                           # ur3 = 0  (drilling)
    A("NX0, 3, 3")
    A("NXA, 3, 3")
    A("NX0, 1, 1")
    A("**")
    A("*STEP, NAME=SINELOAD")
    A("*STATIC")
    A("*DLOAD")
    for i in range(nel):
        xc = (i + 0.5) * a / nel
        A("%d, P, %.6e" % (i + 1, q0 * np.sin(p * xc)))
    A("*OUTPUT, FIELD")
    A("*ELEMENT OUTPUT")
    A("SF, SM")
    A("*NODE OUTPUT")
    A("U")
    A("*NODE PRINT, NSET=NMID")
    A("U")
    A("*EL PRINT, ELSET=EMID")
    A("SF, SM")
    A("*EL PRINT, ELSET=EEND")
    A("SF, SM")
    A("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def write_plate_strip_inp(path, ABDG, a=1.0, q0=1.0e4, nel=100, header_lines=(),
                          coupled=False):
    """The MSG-RM route: the 8x8 plate law installed as a general shell section.

    Variables: ABDG = (8, 8) [[A,B,0],[B,D,0],[0,0,G]] (e.g. rm_plate_msg's
    r["ABDG"]); AB/G2 = its 6x6 and 2x2 blocks; tri = the 21 general-section
    constants (Abaqus order: lower triangle of [[A,B],[B,D]] BY COLUMNS); the
    *TRANSVERSE SHEAR STIFFNESS line is (K11, K22, K12); coupled = the BC mode
    passed through to _strip_inp (True for angle-ply laminates).
    """
    ABDG = np.asarray(ABDG, float)
    if ABDG.shape != (8, 8):
        raise ValueError("ABDG must be 8x8, got %s" % (ABDG.shape,))
    AB = ABDG[:6, :6]
    G2 = ABDG[6:, 6:]
    # 21 constants, Abaqus *SHELL GENERAL SECTION order: lower triangle by columns
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]
    section = ["** ---- the MSG-RM section: [[A,B],[B,D]] 21 constants + the 2x2 shear G ----",
               "*SHELL GENERAL SECTION, ELSET=EALL"]
    section += _fmt_data_lines(tri)
    section += ["*TRANSVERSE SHEAR STIFFNESS",
                "%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1])]
    hdr = list(header_lines) + [
        "D11 = %.6e ; G11 = %.6e ; G22 = %.6e" % (AB[3, 3], G2[0, 0], G2[1, 1])]
    return _strip_inp(path, "Cylindrical-bending strip, MSG-RM 8x8 general shell section",
                      hdr, [], section, a, q0, nel, coupled=coupled)


def write_plate_strip_inp_fsdt(path, thick, angles_deg, mat_names, material_db,
                               a=1.0, q0=1.0e4, nel=100, header_lines=(),
                               coupled=False):
    """The COMMUNITY-FSDT route: ply-by-ply composite section, Abaqus does the rest.

    Materials go in as *ELASTIC, TYPE=LAMINA (E1, E2, nu12, G12, G13, G23) and the
    stack as *SHELL SECTION, COMPOSITE (one line per ply: thickness, integration
    points, material, angle).  Abaqus assembles A/B/D by classical lamination theory
    and computes its OWN transverse-shear stiffness estimate from the layup -- the
    standard practitioner's FSDT shell model, with no MSG content anywhere.

    Variables: used = the distinct material names (one *MATERIAL block each);
    mat_lines/section = the lamina-material and composite-stack deck blocks (one
    ply line = thickness, 3 section points, material, angle); coupled = the BC
    mode passed through to _strip_inp (True for angle-ply laminates).
    """
    used = list(dict.fromkeys(mat_names))
    mat_lines = ["** ---- ply materials (lamina) ----"]
    for m in used:
        E = material_db[m]["E"]; G = material_db[m]["G"]; nu = material_db[m]["nu"]
        mat_lines += ["*MATERIAL, NAME=%s" % m.upper(),
                      "*ELASTIC, TYPE=LAMINA",
                      "%.6e, %.6e, %.6g, %.6e, %.6e, %.6e"
                      % (E[0], E[1], nu[0], G[0], G[1], G[2])]
    section = ["** ---- community FSDT: the ply stack; Abaqus builds ABD + shear ----",
               "*SHELL SECTION, ELSET=EALL, COMPOSITE"]
    for t, ang, m in zip(thick, angles_deg, mat_names):
        section.append("%.8e, 3, %s, %g" % (t, m.upper(), ang))
    return _strip_inp(path, "Cylindrical-bending strip, community FSDT composite section",
                      list(header_lines), mat_lines, section, a, q0, nel,
                      coupled=coupled)
