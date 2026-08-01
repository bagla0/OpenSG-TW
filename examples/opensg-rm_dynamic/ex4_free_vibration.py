"""ex4_free_vibration.py -- Nayak Example 4: natural frequencies of the
CANTILEVERED graphite/epoxy-aluminium sandwich plates measured by Crawley
(their Table 5: experiment, Crawley's FEM, and Nayak's 4-/9-node HSDT FE).

A cantilever has NO Navier solution, so the comparison column here is the
OpenSG-RM route end-to-end: the through-thickness 1-D SG homogenization
(rm_plate_msg, the same call the Abaqus general-section decks use) feeding a
RITZ free-vibration solve of the 5-field RM plate -- no shear correction
factor anywhere, layup-anisotropic D16/D26 included, in-plane/bending
coupling retained (all three layups are symmetric, but nothing assumes it).

Outputs: the comparison table printed + written to ex4_freq_table.dat.

Run:  python examples/opensg-rm_dynamic/ex4_free_vibration.py
"""
import os
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss, Legendre

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

# ----------------------------------------------------------------------------
# ALL VARIABLES (Nayak Example 4 digits, from Crawley's specimens)
# ----------------------------------------------------------------------------
AX = 0.152              # cantilever length a [m] (clamped edge at x = 0)
BY = 0.076              # width b [m]
TPLY = 0.00013          # nominal GE ply thickness [m]
TAL = 0.001             # 2024-T3 aluminium core sheet thickness [m]
MATERIAL_DB = {
    "ge": {"E": [128.0e9, 11.0e9, 11.0e9],      # EL, ET, (ET)
           "G": [4.48e9, 4.48e9, 1.53e9],       # GLT, G13, G23
           "nu": [0.25, 0.25, 0.25], "rho": 1500.0},
    "al": {"E": [68.9e9] * 3,                   # isotropic aluminium
           "G": [68.9e9 / 2.6] * 3,             # G = E/2(1+nu), nu = 0.30
           "nu": [0.30, 0.30, 0.30], "rho": 2770.0},
}
# the three Table-5 layups, bottom -> top (taking the paper's notation
# literally: (0_4/Al)s, (0/+-45/90/Al)s, (+45/-45/Al)s)
LAYUPS = {
    "(0_4/Al)s": ([0, 0, 0, 0, 0, 0, 0, 0, 0.0],
                  ["ge"] * 4 + ["al"] + ["ge"] * 4,
                  [TPLY] * 4 + [TAL] + [TPLY] * 4),
    "(0/+-45/90/Al)s": ([0, 45, -45, 90, 0, 90, -45, 45, 0.0],
                        ["ge"] * 4 + ["al"] + ["ge"] * 4,
                        [TPLY] * 4 + [TAL] + [TPLY] * 4),
    # the paper writes (+-45/Al)s but states "eight plies of GE" for every
    # plate -- with 2 plies/face the frequencies come out ~22 % low, with
    # 4 plies/face they land on Table 5, so the stack is ((+-45)_2/Al)s
    "((+-45)_2/Al)s": ([45, -45, 45, -45, 0, -45, 45, -45, 45.0],
                       ["ge"] * 4 + ["al"] + ["ge"] * 4,
                       [TPLY] * 4 + [TAL] + [TPLY] * 4),
}
# Nayak Table 5 [Hz]: experiment, Crawley FEM, Nayak present-9 (8x4 mesh)
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
NI, NJ = 9, 7           # Ritz polynomial orders (x, y) per field
NMODES = 5              # frequencies reported


def ritz_modes(angles, mats, thick):
    """First NMODES frequencies [Hz] of the cantilevered RM plate whose
    constitutive law is the OpenSG-RM 8x8 of (thick, angles, mats).

    Ritz basis per field: xi * P_i(2 xi - 1) * P_j(2 eta - 1) with
    xi = x/a, eta = y/b -- the xi factor enforces the clamp (all five fields
    zero at x = 0; RM needs nothing else).  Fields d = (u0, v0, w, phx, phy);
    strains E6 = (u0,x; v0,y; u0,y+v0,x; phx,x; phy,y; phx,y+phy,x) against
    the 6x6 in-plane/bending block and gamma = (phx + w,x; phy + w,y)
    against the 2x2 G block.  Mass: I0 (u,v,w) + I2 (phx,phy) + I1 cross."""
    r = rm_plate_msg(thick, angles, mats, MATERIAL_DB, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    A6, G2 = ABDG[:6, :6], ABDG[6:8, 6:8]
    db = MATERIAL_DB
    z = np.concatenate([[0.0], np.cumsum(thick)]) - sum(thick) / 2
    I0 = sum(db[m]["rho"] * (z[k + 1] - z[k])
             for k, m in enumerate(mats))
    I1 = sum(db[m]["rho"] * (z[k + 1] ** 2 - z[k] ** 2) / 2
             for k, m in enumerate(mats))
    I2 = sum(db[m]["rho"] * (z[k + 1] ** 3 - z[k] ** 3) / 3
             for k, m in enumerate(mats))
    nb = NI * NJ                     # basis functions per field
    ntot = 5 * nb
    gx, wx = leggauss(NI + 3)        # Gauss grids: exact for the products
    gy, wy = leggauss(NJ + 3)
    xi = 0.5 * (gx + 1.0)            # map [-1,1] -> [0,1]
    eta = 0.5 * (gy + 1.0)
    wxi, weta = 0.5 * wx, 0.5 * wy
    # basis values and derivatives on the tensor grid, shape (nb, nx, ny)
    Px = [Legendre.basis(i)(2 * xi - 1) for i in range(NI)]
    dPx = [2 * Legendre.basis(i).deriv()(2 * xi - 1) for i in range(NI)]
    Py = [Legendre.basis(j)(2 * eta - 1) for j in range(NJ)]
    dPy = [2 * Legendre.basis(j).deriv()(2 * eta - 1) for j in range(NJ)]
    N = np.empty((nb, len(xi), len(eta)))
    Nx = np.empty_like(N)            # d/dx = (1/a) d/dxi
    Ny = np.empty_like(N)            # d/dy = (1/b) d/deta
    k = 0
    for i in range(NI):
        f, fx = xi * Px[i], Px[i] + xi * dPx[i]   # the clamp factor xi
        for j in range(NJ):
            N[k] = np.outer(f, Py[j])
            Nx[k] = np.outer(fx, Py[j]) / AX
            Ny[k] = np.outer(f, dPy[j]) / BY
            k += 1
    K = np.zeros((ntot, ntot))
    M = np.zeros((ntot, ntot))
    sl = [slice(f * nb, (f + 1) * nb) for f in range(5)]   # u,v,w,phx,phy
    area = AX * BY
    for ix in range(len(xi)):
        for iy in range(len(eta)):
            wq = wxi[ix] * weta[iy] * area
            n, nx, ny = N[:, ix, iy], Nx[:, ix, iy], Ny[:, ix, iy]
            BE = np.zeros((6, ntot))          # E6 rows
            BE[0, sl[0]] = nx                 # e11 = u0,x
            BE[1, sl[1]] = ny                 # e22 = v0,y
            BE[2, sl[0]] = ny                 # g12 = u0,y + v0,x
            BE[2, sl[1]] += nx
            BE[3, sl[3]] = nx                 # k11 = phx,x
            BE[4, sl[4]] = ny                 # k22 = phy,y
            BE[5, sl[3]] = ny                 # k12 = phx,y + phy,x
            BE[5, sl[4]] += nx
            BG = np.zeros((2, ntot))          # transverse shear rows
            BG[0, sl[2]] = nx                 # g13 = w,x + phx
            BG[0, sl[3]] += n
            BG[1, sl[2]] = ny                 # g23 = w,y + phy
            BG[1, sl[4]] += n
            K += wq * (BE.T @ A6 @ BE + BG.T @ G2 @ BG)
            BU = np.zeros((3, ntot))          # translational velocities
            BU[0, sl[0]] = n
            BU[1, sl[1]] = n
            BU[2, sl[2]] = n
            BR = np.zeros((2, ntot))          # rotations phx, phy
            BR[0, sl[3]] = n
            BR[1, sl[4]] = n
            M += wq * (I0 * BU.T @ BU + I2 * BR.T @ BR
                       + I1 * (BU[:2].T @ BR + BR.T @ BU[:2]))
    from scipy.linalg import eigh
    lam = eigh(K, M, eigvals_only=True)
    lam = lam[lam > 1e-3]
    return np.sqrt(lam[:NMODES]) / (2 * np.pi)


def main():
    lines = ["Nayak Ex.4 (Crawley cantilever sandwich): frequencies [Hz]",
             "%-18s %-6s %10s %10s %12s %12s %8s" %
             ("layup", "mode", "Expt[18]", "FEM[18]", "Nayak P9-8x4",
              "OpenSG-RM", "%vsExpt")]
    for name, (angles, mats, thick) in LAYUPS.items():
        f_rm = ritz_modes(angles, mats, thick)
        expt, fem, p9 = TABLE5[name]
        for m in range(NMODES):
            lines.append("%-18s %-6d %10.1f %10.1f %12.1f %12.1f %+8.1f" %
                         (name if m == 0 else "", m + 1, expt[m], fem[m],
                          p9[m], f_rm[m],
                          100 * (f_rm[m] - expt[m]) / expt[m]))
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(HERE, "ex4_freq_table.dat"), "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
