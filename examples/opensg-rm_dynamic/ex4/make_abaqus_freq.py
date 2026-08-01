"""make_abaqus_freq.py -- Nayak Example 4: the OpenSG-RM free-vibration
route, done the same way as Ex.5 -- ABAQUS does the analysis, the MSG ABDG
is the only constitutive input.

PROBLEM STATEMENT (Nayak Ex.4 = Crawley's experiment): a CANTILEVERED
rectangular sandwich plate, clamped along the short edge x = 0, free on the
other three edges; a = 0.152 m (length) x b = 0.076 m (width).  Eight
graphite/epoxy plies (EL = 128, ET = 11.0, GLT = G13 = 4.48, G23 = 1.53 GPa,
nu = 0.25, rho = 1500 kg/m^3, nominal ply 0.13 mm) laid symmetrically about
a 1 mm sheet of 2024-T3 aluminium (E = 68.9 GPa, nu = 0.30,
rho = 2770 kg/m^3).  Three stacks: (0_4/Al)s, (0/+-45/90/Al)s and
((+-45)_2/Al)s [the paper prints "(+-45/Al)s" but states EIGHT GE plies --
4 per face lands on Table 5, 2 per face comes out ~22 % low].
Wanted: the first five natural frequencies, against their Table 5 --
Crawley's EXPERIMENT, Crawley's FEM, and Nayak's Reddy-HSDT FE.

THE OpenSG-RM ROUTE (per layup):
  1. the layup becomes a through-thickness 1-D SG (ex4_<slug>_sg.yaml);
  2. rm_plate_msg homogenizes it to the RM 8x8 ABDG;
  3. THIS script writes a plain Abaqus S4 deck whose ONLY constitutive
     input is that ABDG: the 6x6 goes in as *SHELL GENERAL SECTION
     (DENSITY = rho h) and the 2x2 shear block as *TRANSVERSE SHEAR
     STIFFNESS (the +-45 stacks carry a nonzero G12 coupling term);
  4. Abaqus runs *FREQUENCY (Lanczos, 8 modes requested, 5 reported) --
     a standard 2-D shell vibration analysis, no plies, no orientations,
     no shear correction factor anywhere.

Deck details: 24 x 12 S4 mesh (6.3 mm elements), ENCASTRE on the x = 0
node row, drilling dof 6 fixed everywhere (flat plate).  After running the
three jobs (ABAQUS_RUN_COMMANDS.md) copy ex4_*_freq.dat into
Abaqus_results/ and run collect_freq.py for the table + figures.

Run:  python examples/opensg-rm_dynamic/ex4/make_abaqus_freq.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg
from opensg_jax.fe_jax.segment_plate import plate_sg_yaml, plot_plate_sg

# ----------------------------------------------------------------------------
# ALL CASE DATA (Nayak Ex.4 digits; shared with collect_freq.py)
# ----------------------------------------------------------------------------
AX = 0.152              # cantilever length a [m] (clamped edge at x = 0)
BY = 0.076              # width b [m]
TPLY = 0.00013          # nominal GE ply thickness [m]
TAL = 0.001             # 2024-T3 aluminium core sheet thickness [m]
NEX, NEY = 24, 12       # S4 mesh (6.3 mm elements)
NMODES = 5              # frequencies reported (8 extracted)
MATERIAL_DB = {
    "ge": {"E": [128.0e9, 11.0e9, 11.0e9],
           "G": [4.48e9, 4.48e9, 1.53e9],
           "nu": [0.25, 0.25, 0.25], "rho": 1500.0},
    "al": {"E": [68.9e9] * 3, "G": [68.9e9 / 2.6] * 3,
           "nu": [0.30, 0.30, 0.30], "rho": 2770.0},
}
LAYUPS = {
    "(0_4/Al)s": ([0, 0, 0, 0, 0, 0, 0, 0, 0.0],
                  ["ge"] * 4 + ["al"] + ["ge"] * 4,
                  [TPLY] * 4 + [TAL] + [TPLY] * 4),
    "(0/+-45/90/Al)s": ([0, 45, -45, 90, 0, 90, -45, 45, 0.0],
                        ["ge"] * 4 + ["al"] + ["ge"] * 4,
                        [TPLY] * 4 + [TAL] + [TPLY] * 4),
    "((+-45)_2/Al)s": ([45, -45, 45, -45, 0, -45, 45, -45, 45.0],
                       ["ge"] * 4 + ["al"] + ["ge"] * 4,
                       [TPLY] * 4 + [TAL] + [TPLY] * 4),
}
SLUGS = {"(0_4/Al)s": "04_Al", "(0/+-45/90/Al)s": "0_pm45_90_Al",
         "((+-45)_2/Al)s": "pm45x2_Al"}
# Nayak Table 5 [Hz]: experiment (Crawley), Crawley's FEM, and Nayak's
# Reddy-HSDT 9-node element on the 8x4 mesh (his most converged column)
TABLE5 = {
    "(0_4/Al)s": ([101.7, 229.0, 631.9, 865.0, 1129.0],
                  [108.8, 228.8, 680.2, 885.6, 1168.0],
                  [108.2, 227.3, 675.0, 879.6, 1147.4]),
    "(0/+-45/90/Al)s": ([75.9, 302.0, 469.6, 983.0, 1306.0],
                        [81.16, 313.8, 505.1, 1035.0, 1438.0],
                        [80.0, 311.9, 501.0, 1028.3, 1399.8]),
    "((+-45)_2/Al)s": ([58.3, 351.6, 358.0, 1006.0, 1113.0],
                       [58.46, 354.70, 379.60, 1029.0, 1187.0],
                       [57.9, 352.8, 377.4, 1020.6, 1179.2]),
}


def write_freq_inp(slug, angles, mats, thick):
    """One Abaqus *FREQUENCY deck: S4 mesh + the MSG ABDG section + the
    x = 0 clamp.  Returns the deck path."""
    r = rm_plate_msg(list(thick), list(map(float, angles)), list(mats),
                     MATERIAL_DB, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    AB, G2 = ABDG[:6, :6], ABDG[6:8, 6:8]
    rho_h = sum(MATERIAL_DB[m]["rho"] * t for m, t in zip(mats, thick))
    tri = [AB[i, j] for j in range(6) for i in range(j + 1)]  # upper tri
    dx, dy = AX / NEX, BY / NEY

    def n(i, j):
        return 1 + i + (NEX + 1) * j

    L = ["*HEADING",
         "Nayak Ex.4 cantilever sandwich %s -- OpenSG-RM (MSG ABDG) S4"
         " frequency extraction" % slug,
         "** a=%.4g b=%.4g rho_h=%.6g kg/m^2" % (AX, BY, rho_h),
         "*NODE"]
    for j in range(NEY + 1):
        for i in range(NEX + 1):
            L.append("%d, %.8f, %.8f, 0.0" % (n(i, j), i * dx, j * dy))
    L.append("*ELEMENT, TYPE=S4, ELSET=EALL")
    for j in range(NEY):
        for i in range(NEX):
            L.append("%d, %d, %d, %d, %d" % (1 + i + NEX * j, n(i, j),
                                             n(i + 1, j), n(i + 1, j + 1),
                                             n(i, j + 1)))
    L.append("*NSET, NSET=NROOT")
    ids = [n(0, j) for j in range(NEY + 1)]
    for s in range(0, len(ids), 12):
        L.append(", ".join(str(v) for v in ids[s:s + 12]))
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append("1, %d, 1" % n(NEX, NEY))
    L.append("*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=%.6g" % rho_h)
    for s in range(0, len(tri), 8):
        L.append(", ".join("%.6e" % v for v in tri[s:s + 8]))
    L.append("*TRANSVERSE SHEAR STIFFNESS")
    L.append("%.6e, %.6e, %.6e" % (G2[0, 0], G2[1, 1], G2[0, 1]))
    L.append("** clamp the x = 0 edge; drilling dof fixed (flat plate)")
    L.append("*BOUNDARY")
    L.append("NROOT, 1, 6")
    L.append("NALL, 6, 6")
    L.append("*STEP, NAME=MODES")
    L.append("*FREQUENCY")
    L.append("8,")
    L.append("*END STEP")
    path = os.path.join(HERE, "ex4_%s_freq.inp" % slug)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path, rho_h


def main():
    for name, (angles, mats, thick) in LAYUPS.items():
        slug = SLUGS[name]
        yml = os.path.join(HERE, "ex4_%s_sg.yaml" % slug)
        plate_sg_yaml(yml, {"mat_names": list(mats), "thick": list(thick),
                            "angles": list(map(float, angles))},
                      MATERIAL_DB, fraction=0.5)
        plot_plate_sg(yml)
        path, rho_h = write_freq_inp(slug, angles, mats, thick)
        print("wrote %s (rho_h = %.4f kg/m^2)"
              % (os.path.basename(path), rho_h))
    print("run the three jobs on the Abaqus machine, copy ex4_*_freq.dat"
          " into Abaqus_results/, then run collect_freq.py")


if __name__ == "__main__":
    main()
