"""compare_ex3.py -- Nayak Example 3 table: OpenSG-RM (Abaqus S4 static +
SG recovery) against PAGANO's exact 3-D elasticity values and Nayak's
9-node FE (13 nodes per side on the quarter plate) -- both sets of digits
taken directly from his Table 4.

Chain per a/h: parse Abaqus_results/ex3_RM_S<S>.dat -> quadratic patch fit
of SF/SM about each station -> E6 = S6 R6 + gradients + the static
double-sine load ladder -> msgrm_strain_at_depth at the tabulated (x,y,z)
-> Nayak's normalizations: in-plane x h^2/(q0 a^2), shear x h/(q0 a).

Output: ex3_table.dat -- Exact [Pagano] | Nayak P9 (13 nodes) | OpenSG-RM
with percent errors vs the exact values.

Run:  python examples/opensg-rm_dynamic/ex3/compare_ex3.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "yu2003"))

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from recover_6p2 import read_elprint_tables                     # noqa: E402

A = 1.0
NX = 20
Q0 = 1.0e4
P = np.pi / A
DX = A / NX
# Nayak Table 4 digits: (sx, sy, sxy, sxz, syz) normalized
TAB4 = {
    10: {"exact": [1.153, 0.1104, 0.0707, 0.3000, 0.0527],
         "nayak": [1.151, 0.1043, 0.0689, 0.3506, 0.0580]},
    100: {"exact": [1.098, 0.0550, 0.0437, 0.3240, 0.0297],
          "nayak": [1.099, 0.0550, 0.0437, 0.3760, 0.0343]},
}
LBL = ["sx(a/2,a/2,h/2)", "sy(a/2,a/2,h/2)", "sxy(0,0,h/2)",
       "sxz(0,a/2,0)", "syz(a/2,0,0)"]


def patch_fit(tables, name, x0):
    """(8, 3) fitted resultants + gradients about x0 from one patch."""
    sf = None; xy = None
    for (es, labels), rows in tables.items():
        if es != name:
            continue
        ofs = rows.shape[1] - len(labels)
        if "SF1" in labels:
            idx = [labels.index(k) + ofs for k in
                   ("SF1", "SF2", "SF3", "SF4", "SF5", "SM1", "SM2", "SM3")]
            sf = rows[:16, idx]
        if "COORD1" in labels:
            idx = [labels.index(k) + ofs for k in ("COORD1", "COORD2")]
            xy = rows[:16, idx]
    xi, eta = xy[:, 0] - x0[0], xy[:, 1] - x0[1]
    B = np.column_stack([np.ones(16), xi, eta, xi * eta, xi ** 2, eta ** 2])
    C, *_ = np.linalg.lstsq(B, sf, rcond=None)
    return np.column_stack([C[0], C[1], C[2]])


def ladder(x, y):
    s1, c1 = np.sin(P * x), np.cos(P * x)
    s2, c2 = np.sin(P * y), np.cos(P * y)
    return Q0 * np.array([s1 * s2, P * c1 * s2, P * s1 * c2,
                          -P * P * s1 * s2, P * P * c1 * c2,
                          -P * P * s1 * s2])


def recover_at(r, S6, fits, st, x0, z, comp):
    """One recovered Voigt component at station st, height z."""
    F = fits[st]
    E6 = S6 @ F[[0, 1, 2, 5, 6, 7], 0]
    dE1 = S6 @ F[[0, 1, 2, 5, 6, 7], 1]
    dE2 = S6 @ F[[0, 1, 2, 5, 6, 7], 2]
    z6 = np.zeros(6)
    sig = msgrm_strain_at_depth(r, z, E6, dE1, dE2, -P * P * E6, z6,
                                -P * P * E6, qt6=ladder(*x0))[1]
    return sig[comp]


def run(S):
    dat = os.path.join(HERE, "Abaqus_results", "ex3_RM_S%d.dat" % S)
    if not os.path.isfile(dat):
        print("missing %s -- run the Abaqus job first" % dat)
        return None
    h = A / S
    yml = os.path.join(HERE, "ex3_S%d_sg.yaml" % S)
    inp = read_plate_sg_yaml(yml)
    r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                     inp["material_db"], fraction=inp["fraction"])
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    tables = read_elprint_tables(dat)
    STA = {"C": (A / 2, A / 2), "O": (0.0, 0.0),
           "X": (0.0, A / 2), "Y": (A / 2, 0.0)}
    fits = {st: patch_fit(tables, "PATCH" + st, STA[st]) for st in STA}
    zt = h / 2 - 1e-9
    # Voigt order of the recovery: (11, 22, 33, 23, 13, 12)
    vals = [
        recover_at(r, S6, fits, "C", STA["C"], zt, 0),    # sx  top center
        recover_at(r, S6, fits, "C", STA["C"], zt, 1),    # sy  top center
        recover_at(r, S6, fits, "O", STA["O"], zt, 5),    # sxy top corner
        recover_at(r, S6, fits, "X", STA["X"], 0.0, 4),   # sxz mid, x-edge
        recover_at(r, S6, fits, "Y", STA["Y"], 0.0, 3),   # syz mid, y-edge
    ]
    nin = h * h / (Q0 * A * A)                 # in-plane normalization
    nsh = h / (Q0 * A)                         # transverse-shear norm.
    return [abs(vals[0]) * nin, abs(vals[1]) * nin, abs(vals[2]) * nin,
            abs(vals[3]) * nsh, abs(vals[4]) * nsh]


def main():
    lines = ["Nayak Ex.3 / Pagano sandwich (0/core/0), sinsin static load",
             "columns: Exact 3-D elasticity [Pagano] | Nayak 9-node FE"
             " (13 nodes/side, Table 4) | OpenSG-RM (Abaqus S4 static +"
             " SG recovery); %err vs Exact"]
    for S in (10, 100):
        rm = run(S)
        if rm is None:
            continue
        ex, ny = TAB4[S]["exact"], TAB4[S]["nayak"]
        lines += ["", "a/h = %d:" % S,
                  "%-18s %10s %12s %12s %12s %12s" %
                  ("quantity", "Exact", "Nayak-P9", "OpenSG-RM",
                   "%err(Nayak)", "%err(RM)")]
        for k in range(5):
            lines.append("%-18s %10.4f %12.4f %12.4f %+12.2f %+12.2f" %
                         (LBL[k], ex[k], ny[k], rm[k],
                          100 * (ny[k] - ex[k]) / ex[k],
                          100 * (rm[k] - ex[k]) / ex[k]))
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(HERE, "ex3_table.dat"), "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
