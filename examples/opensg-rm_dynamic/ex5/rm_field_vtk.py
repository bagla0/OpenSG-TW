"""rm_field_vtk.py -- STANDALONE: the full 3-D OpenSG-RM field as a VTK
volume, dehomogenized at EVERY global plate node and every through-thickness
station, from nothing but the Abaqus 2-D shell .dat.

Chain (all in this one file):
  1. parse Abaqus_results/sandwich_RM_field.dat: all 441 nodal U and all
     400-element SF/SM of the whole-plate dumps (every 57th increment);
     pick the dump with the largest center |w| (the 2.85 ms peak);
  2. element resultants -> nodal 21x21 grids (adjacent-cell averaging),
     in-plane gradients by finite differences on the grid;
  3. drivers per node: E6 = S6 R6, dE1/dE2 (FD), the second gradients
     (FD of the gradients), and the closed-form double-sine load ladder
     qt6 of the applied face pressure at the snapshot instant;
  4. the recovery msgrm_strain_at_depth / msgrm_warping_at_depth is LINEAR
     in those 42 drivers, so the through-thickness operators are built once
     per z (42 unit evaluations) and the whole plate is one contraction --
     evaluated with jax.vmap over the global nodes;
  5. the sigma_a3 columns are rescaled per node to carry the plate's own
     Q1/Q2 (the dynamic-consistency rule of recover_dyn.py), and the 3-D
     displacement is composed as U_a = u_a - z w,_a + warp_a, U3 = w + w3;
  6. write sandwich_rm_field.vtk: STRUCTURED_GRID 21x21xNZ with the
     displacement vector + all six stress components as point data --
     contour it over the 3-D geometry in ParaView (no mesh lines).

Run:  python examples/opensg-rm_dynamic/ex5/rm_field_vtk.py
"""
import os
import re
import sys

import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "yu2003"))

from opensg_jax.fe_jax.segment_plate import read_plate_sg_yaml
from opensg_jax.fe_jax.msg_rm_plate import (rm_plate_msg,
                                            msgrm_strain_at_depth,
                                            msgrm_warping_at_depth)
from recover_6p2 import read_elprint_tables                     # noqa: E402

A = 1.524
NX = 20
Q0 = 68.9476e6
DT = 5.0e-5
P = np.pi / A
NPZ = [4, 4, 4, 4, 12, 4, 4, 4, 4]      # z stations per layer (44 total)

inp = read_plate_sg_yaml(os.path.join(HERE, "sandwich_sg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
S6 = np.linalg.inv(np.asarray(r["A6"]))
thick = inp["thick"]
H = float(sum(thick))


def parse_field_dat(path):
    """All whole-plate dumps: nodal U (n_dump, 441, 6) and element SF/SM
    (n_dump, 400, 8) (integration points averaged per element)."""
    rows = []
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    active, labels_seen = False, False
    for ln in lines:
        if "NODE SET NALL" in ln and "TABLE IS PRINTED" in ln:
            active, labels_seen = True, False
            continue
        toks = ln.split()
        if not toks:
            continue
        if active and not labels_seen:
            if toks[0] == "NODE":
                labels_seen = True
            continue
        if active and labels_seen:
            if re.fullmatch(r"\d+", toks[0]):
                vals = [float(toks[0])]
                for t in toks[1:]:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        pass
                rows.append(vals)
            elif toks[0] in ("MAXIMUM", "MINIMUM"):
                active = False
    rows = np.array(rows)                      # (n_dump*441, 7)
    nn = (NX + 1) * (NX + 1)
    U = rows[:, 1:7].reshape(-1, nn, 6)
    order = np.argsort(rows[:U.shape[1], 0])   # ids are already sorted
    U = U[:, order]
    t = read_elprint_tables(path)
    SF = None
    for (es, labels), rws in t.items():
        if es == "EALL" and "SF1" in labels:
            ofs = rws.shape[1] - len(labels)
            idx = [labels.index(k) + ofs for k in
                   ("SF1", "SF2", "SF3", "SF4", "SF5", "SM1", "SM2", "SM3")]
            v = rws[:, idx].reshape(-1, NX * NX, 4, 8)   # per-elem 4 pts
            SF = v.mean(axis=2)
    return U, SF


def nodal_from_elements(F):
    """(400, k) element-centroid fields -> (21, 21, k) nodal grids by
    averaging the (up to four) adjacent cells."""
    k = F.shape[1]
    Fe = F.reshape(NX, NX, k)                 # [j, i] row-major (j outer)
    G = np.zeros((NX + 1, NX + 1, k))
    Wt = np.zeros((NX + 1, NX + 1, 1))
    for dj in (0, 1):
        for di in (0, 1):
            G[dj:dj + NX, di:di + NX] += Fe
            Wt[dj:dj + NX, di:di + NX] += 1.0
    return G / Wt


def main():
    dat = os.path.join(HERE, "Abaqus_results", "sandwich_RM_field.dat")
    U, SF = parse_field_dat(dat)
    icen = (NX // 2) * (NX + 1) + NX // 2      # node (10,10)
    kd = int(np.argmax(np.abs(U[:, icen, 2])))
    tstar = 57 * DT * (kd + 1)
    print("using dump %d (t = %.2f ms), center w = %.4f m"
          % (kd + 1, 1e3 * tstar, U[kd, icen, 2]))
    dx = A / NX
    ug = U[kd, :, :3].reshape(NX + 1, NX + 1, 3)      # [j, i] grid, u v w
    Rg = nodal_from_elements(SF[kd])                  # (21,21,8) resultants
    # in-plane gradients on the grid: axis0 = y (j), axis1 = x (i)
    def grad(Fg):
        gy, gx = np.gradient(Fg, dx, dx, axis=(0, 1))
        return gx, gy

    E6 = np.einsum("ab,ijb->ija", S6, Rg[:, :, [0, 1, 2, 5, 6, 7]])
    dE1, dE2 = grad(E6)
    dE11, _ = grad(dE1)
    dE12a, dE22 = grad(dE2)
    Q1, Q2 = Rg[:, :, 3], Rg[:, :, 4]
    wx, wy = grad(ug[:, :, 2:3])
    wx, wy = wx[:, :, 0], wy[:, :, 0]
    # the closed-form load ladder of q = Q0 sin sin at the snapshot (F=1)
    xi = np.arange(NX + 1) * dx
    Xg, Yg = np.meshgrid(xi, xi)               # [j, i]: X varies along i
    s1, c1 = np.sin(P * Xg), np.cos(P * Xg)
    s2, c2 = np.sin(P * Yg), np.cos(P * Yg)
    qt6 = Q0 * np.stack([s1 * s2, P * c1 * s2, P * s1 * c2,
                         -P * P * s1 * s2, P * P * c1 * c2,
                         -P * P * s1 * s2], axis=-1)
    # driver vector per node: 42 = 6 blocks of E-drivers + the ladder
    D = np.concatenate([E6, dE1, dE2, dE11, dE12a, dE22, qt6],
                       axis=-1).reshape(-1, 42)
    # ---- through-thickness operators: recovery is LINEAR in the drivers
    zk = np.concatenate([[0.0], np.cumsum(thick)]) - H / 2
    zg = np.concatenate([np.linspace(zk[m] + 1e-9, zk[m + 1] - 1e-9, npz)
                         for m, npz in enumerate(NPZ)])
    nz = len(zg)
    TS = np.zeros((nz, 6, 42))                 # stress operator
    TW = np.zeros((nz, 3, 42))                 # warping-displacement op.
    z6 = np.zeros(6)
    for iz, z in enumerate(zg):
        for j in range(42):
            d = np.zeros(42)
            d[j] = 1.0
            e6, g1, g2, g11, g12, g22, q6 = (d[0:6], d[6:12], d[12:18],
                                             d[18:24], d[24:30], d[30:36],
                                             d[36:42])
            TS[iz, :, j] = msgrm_strain_at_depth(
                r, z, e6, g1, g2, g11, g12, g22, qt6=q6)[1]
            TW[iz, :, j] = msgrm_warping_at_depth(
                r, z, e6, g1, g2, g11, g12, g22, qt6=q6)
    # ---- one contraction for the whole plate: jax.vmap over the nodes
    TSj, TWj = jnp.asarray(TS), jnp.asarray(TW)

    @jax.jit
    def node_field(d):
        return jnp.einsum("zcj,j->zc", TSj, d), \
               jnp.einsum("zcj,j->zc", TWj, d)

    Sig, Warp = jax.vmap(node_field)(jnp.asarray(D))
    Sig, Warp = np.array(Sig), np.array(Warp)       # writable copies
    # ---- dynamic-consistency rescale of the sigma_a3 columns per node
    I13 = np.trapezoid(Sig[:, :, 4], zg, axis=1)
    I23 = np.trapezoid(Sig[:, :, 3], zg, axis=1)
    q1, q2 = Q1.reshape(-1), Q2.reshape(-1)
    sc13 = np.where(np.abs(I13) > 1e-3 * np.abs(q1).max(), q1 / I13, 1.0)
    sc23 = np.where(np.abs(I23) > 1e-3 * np.abs(q2).max(), q2 / I23, 1.0)
    Sig[:, :, 4] *= sc13[:, None]
    Sig[:, :, 3] *= sc23[:, None]
    # ---- 3-D displacement: U_a = u_a - z w,_a + warp_a ; U3 = w + w3
    u0 = ug.reshape(-1, 3)
    wxf, wyf = wx.reshape(-1), wy.reshape(-1)
    U1 = u0[:, 0:1] - zg[None, :] * wxf[:, None] + Warp[:, :, 0]
    U2 = u0[:, 1:2] - zg[None, :] * wyf[:, None] + Warp[:, :, 1]
    U3 = u0[:, 2:3] + Warp[:, :, 2]
    # ---- legacy VTK structured grid: 21 x 21 x nz --------------------
    out = os.path.join(HERE, "sandwich_rm_field.vtk")
    nn = (NX + 1) * (NX + 1)
    with open(out, "w") as f:
        f.write("# vtk DataFile Version 3.0\n"
                "OpenSG-RM dehomogenized field, Nayak Ex.5 step pulse,"
                " t = %.4f s\nASCII\nDATASET STRUCTURED_GRID\n"
                "DIMENSIONS %d %d %d\nPOINTS %d float\n"
                % (tstar, NX + 1, NX + 1, nz, nn * nz))
        for iz in range(nz):
            for jj in range(NX + 1):
                for ii in range(NX + 1):
                    f.write("%.6e %.6e %.6e\n"
                            % (ii * dx, jj * dx, zg[iz] + H / 2))
        f.write("POINT_DATA %d\nVECTORS disp float\n" % (nn * nz))
        for iz in range(nz):
            for n in range(nn):
                f.write("%.6e %.6e %.6e\n" % (U1[n, iz], U2[n, iz],
                                              U3[n, iz]))
        comp = {"S11": 0, "S22": 1, "S33": 2, "S23": 3, "S13": 4, "S12": 5}
        for nm, c in comp.items():
            f.write("SCALARS %s float 1\nLOOKUP_TABLE default\n" % nm)
            for iz in range(nz):
                for n in range(nn):
                    f.write("%.6e\n" % Sig[n, iz, c])
    print("wrote %s  (21x21x%d points)" % (os.path.basename(out), nz))
    for nm, c in (("S11", 0), ("S13", 4), ("S23", 3), ("S33", 2)):
        print("  max |%s| = %.4e Pa" % (nm, np.abs(Sig[:, :, c]).max()))


if __name__ == "__main__":
    main()
