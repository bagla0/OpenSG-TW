"""helper_inp_plate.py -- Abaqus .inp GENERATOR for a plate strip carrying the MSG-RM
8x8 section (the plate-solver side of the homo -> plate solve -> FF -> dehom loop).

Writes a complete, submit-ready input file for the cylindrical-bending strip:

  mesh       0 <= x <= a, one square S4 element wide, ``nel`` elements along the span,
             reference surface = the surface the 8x8 was homogenized at
  section    *SHELL GENERAL SECTION with the 21 constants of [[A,B],[B,D]] (Abaqus
             lower-triangle order) + *TRANSVERSE SHEAR STIFFNESS with the MSG 2x2 G
  load       q(x) = q0 sin(pi x / a) on the face, piecewise-constant per element
             (element-centre value; quadrature error ~ (pi/nel)^2 / 24)
  BCs        cylindrical bending (u2 = ur1 = ur3 = 0 everywhere) + simple supports
             (u3 = 0 on both end edges, u1 = 0 at x = 0)
  output     field SF/SM/U to the .odb; .dat prints of U at the mid-span node (NMID)
             and SF/SM in the mid-span (EMID) and end (EEND) elements -- the two
             FF-extraction stations (M max at a/2, Q max at 0)

VALIDATED against a real Abaqus 2024 run (Garg caseA, S = 10): mid-span U3 came back
94.3695e-6 m vs the closed-form q0/(p^4 D11) + q0/(p^2 G11) = 94.3817e-6 (0.013%, FE
discretisation); IP resultants M11 = 1012.92 vs q0/p^2 = 1013.21, Q1 = 3182.44 vs
q0/p = 3183.10.

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


def write_plate_strip_inp(path, ABDG, a=1.0, q0=1.0e4, nel=100, header_lines=()):
    """Write the cylindrical-bending strip .inp for an 8x8 plate law.

    Parameters
    ----------
    path         output .inp path
    ABDG         (8, 8) plate law [[A,B,0],[B,D,0],[0,0,G]] (e.g. rm_plate_msg's r["ABDG"])
    a            span [m]
    q0           load amplitude [Pa]
    nel          elements along the span
    header_lines extra ``**`` comment lines for the file header (e.g. the case, the
                 closed-form deflection predictions)

    Returns the path written.
    """
    ABDG = np.asarray(ABDG, float)
    if ABDG.shape != (8, 8):
        raise ValueError("ABDG must be 8x8, got %s" % (ABDG.shape,))
    AB = ABDG[:6, :6]
    G2 = ABDG[6:, 6:]
    p = np.pi / a
    b = a / nel                                   # one square element across the width
    # 21 constants, Abaqus *SHELL GENERAL SECTION order: lower triangle by columns
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]

    L = []
    A = L.append
    A("*HEADING")
    A("Cylindrical-bending strip, MSG-RM 8x8 general shell section")
    for ln in header_lines:
        A("** " + ln)
    A("** span a = %g m, q0 = %g Pa, %d S4 elements; load q0*sin(pi*x/a) per element" % (a, q0, nel))
    A("** D11 = %.6e ; G11 = %.6e ; G22 = %.6e" % (AB[3, 3], G2[0, 0], G2[1, 1]))
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
    A("*NSET, NSET=NX0")
    A("%d, %d" % (nid(0, 0), nid(0, 1)))
    A("*NSET, NSET=NXA")
    A("%d, %d" % (nid(nel, 0), nid(nel, 1)))
    A("*NSET, NSET=NMID")
    A("%d" % nid(nel // 2, 0))
    A("*NSET, NSET=NALL, GENERATE")
    A("1, %d, 1" % nid(nel, 1))
    A("**")
    A("** ---- the MSG-RM section: [[A,B],[B,D]] 21 constants + the 2x2 shear G ----")
    A("*SHELL GENERAL SECTION, ELSET=EALL")
    L.extend(_fmt_data_lines(tri))
    A("*TRANSVERSE SHEAR STIFFNESS")
    A("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    A("**")
    A("** ---- cylindrical bending + simple supports ----")
    A("*BOUNDARY")
    A("NALL, 2, 2")                               # u2 = 0   (nothing varies with x2)
    A("NALL, 4, 4")                               # ur1 = 0  (kappa22 = 0)
    A("NALL, 6, 6")                               # ur3 = 0  (drilling)
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
