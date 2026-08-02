"""rm_field_vtk_ex2.py -- Ex.2: the full 3-D OpenSG-RM field as a VTK
volume, dehomogenized at EVERY plate node and thickness station from the
Abaqus 2-D shell .dat alone (ex2_RM_field.dat), on the SAME 40x40x36
Gauss lattice as the solid converter -- the ex5 pipeline ported to the
(0/90/0) a = 5h plate, step pulse, snapshot at the 0.75 ms first peak.

Chain (identical logic to ex5/rm_field_vtk.py):
  parse whole-plate dumps -> element SF/SM -> nodal grids + FD gradients
  (+ mode-consistent second gradients, closed-form load ladders) ->
  the recovery is LINEAR in the 42 drivers -> per-z operators once ->
  jax.vmap over the lattice -> per-node Q-rescale of sigma_a3 ->
  sigma33 by the dynamic momentum integral -> ex2_rm_field.vtk.

Run:  python examples/opensg-rm_dynamic/ex2/rm_field_vtk_ex2.py
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

A, H = 0.762, 0.1524
NX, NZT = 20, 18
Q0 = 68.9476e6
DT = 5.0e-5
P = np.pi / A
GA = 0.5 / np.sqrt(3.0)

inp = read_plate_sg_yaml(os.path.join(HERE, "ex2_sg.yaml"))
r = rm_plate_msg(inp["thick"], inp["angles"], inp["mat_names"],
                 inp["material_db"], fraction=inp["fraction"])
S6 = np.linalg.inv(np.asarray(r["A6"]))


def parse_field_dat(path):
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
    rows = np.array(rows)
    nn = (NX + 1) * (NX + 1)
    U = rows[:, 1:7].reshape(-1, nn, 6)
    t = read_elprint_tables(path)
    SF = None
    for (es, labels), rws in t.items():
        if es == "EALL" and "SF1" in labels:
            ofs = rws.shape[1] - len(labels)
            idx = [labels.index(k) + ofs for k in
                   ("SF1", "SF2", "SF3", "SF4", "SF5", "SM1", "SM2", "SM3")]
            SF = rws[:, idx].reshape(-1, NX * NX, 4, 8).mean(axis=2)
    return U, SF


def center_w_history(dat_path):
    rows = []
    with open(dat_path, errors="replace") as f:
        lines = f.read().splitlines()
    active, labels_seen = False, False
    for ln in lines:
        if "NODE SET NCEN" in ln and "TABLE IS PRINTED" in ln:
            active, labels_seen = True, False
            continue
        toks = ln.split()
        if not toks:
            continue
        if active and not labels_seen:
            if toks[0] == "NODE":
                labels_seen = True
            continue
        if active and labels_seen and re.fullmatch(r"\d+", toks[0]):
            vals = []
            for t in toks[1:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if len(vals) >= 3:
                rows.append(vals[2])
            active = False
    return np.array(rows)


def nodal_from_elements(F):
    k = F.shape[1]
    Fe = F.reshape(NX, NX, k)
    Gr = np.zeros((NX + 1, NX + 1, k))
    Wt = np.zeros((NX + 1, NX + 1, 1))
    for dj in (0, 1):
        for di in (0, 1):
            Gr[dj:dj + NX, di:di + NX] += Fe
            Wt[dj:dj + NX, di:di + NX] += 1.0
    return Gr / Wt


def main():
    dat = os.path.join(HERE, "Abaqus_results", "ex2_RM_field.dat")
    U, SF = parse_field_dat(dat)
    icen = (NX // 2) * (NX + 1) + NX // 2
    kd = int(np.argmax(np.abs(U[:, icen, 2])))
    tstar = 15 * DT * (kd + 1)
    print("using dump %d (t = %.3f ms), center w = %.5f m"
          % (kd + 1, 1e3 * tstar, U[kd, icen, 2]))
    dx = A / NX
    ug = U[kd, :, :3].reshape(NX + 1, NX + 1, 3)
    Rg = nodal_from_elements(SF[kd])

    def grad(F):
        gy, gx = np.gradient(F, dx, dx, axis=(0, 1))
        return gx, gy

    E6 = np.einsum("ab,ijb->ija", S6, Rg[:, :, [0, 1, 2, 5, 6, 7]])
    dE1, dE2 = grad(E6)
    dE11 = -P * P * E6
    dE22 = -P * P * E6
    dE12a, _ = grad(dE2)
    Q1, Q2 = Rg[:, :, 3], Rg[:, :, 4]
    wx, wy = grad(ug[:, :, 2:3])
    wx, wy = wx[:, :, 0], wy[:, :, 0]
    # ---- resample onto the solid's Gauss lattice -----------------------
    xg = (np.repeat(np.arange(NX), 2) + 0.5 + np.tile([-GA, +GA], NX)) * dx
    ngx = len(xg)

    def bilerp(F):
        f = xg / dx
        i0 = np.clip(f.astype(int), 0, NX - 1)
        t = f - i0
        Fi = ((1 - t)[None, :, None] * F[:, i0]
              + t[None, :, None] * F[:, i0 + 1])
        Fo = ((1 - t)[:, None, None] * Fi[i0]
              + t[:, None, None] * Fi[i0 + 1])
        return Fo

    E6, dE1, dE2 = bilerp(E6), bilerp(dE1), bilerp(dE2)
    dE11, dE12a, dE22 = bilerp(dE11), bilerp(dE12a), bilerp(dE22)
    Q1 = bilerp(Q1[:, :, None])[:, :, 0]
    Q2 = bilerp(Q2[:, :, None])[:, :, 0]
    u0g = bilerp(ug)
    wx = bilerp(wx[:, :, None])[:, :, 0]
    wy = bilerp(wy[:, :, None])[:, :, 0]
    Xg, Yg = np.meshgrid(xg, xg)
    s1, c1 = np.sin(P * Xg), np.cos(P * Xg)
    s2, c2 = np.sin(P * Yg), np.cos(P * Yg)
    qt6 = Q0 * np.stack([s1 * s2, P * c1 * s2, P * s1 * c2,
                         -P * P * s1 * s2, P * P * c1 * c2,
                         -P * P * s1 * s2], axis=-1)
    D = np.concatenate([E6, dE1, dE2, dE11, dE12a, dE22, qt6],
                       axis=-1).reshape(-1, 42)
    # uniform 18-layer solid mesh -> uniform Gauss depths
    dz = H / NZT
    zg = (np.repeat(np.arange(NZT), 2) + 0.5
          + np.tile([-GA, +GA], NZT)) * dz - H / 2
    nz = len(zg)
    TS = np.zeros((nz, 6, 42))
    TW = np.zeros((nz, 3, 42))
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
    TSj, TWj = jnp.asarray(TS), jnp.asarray(TW)

    @jax.jit
    def node_field(d):
        return jnp.einsum("zcj,j->zc", TSj, d), \
               jnp.einsum("zcj,j->zc", TWj, d)

    Sig, Warp = jax.vmap(node_field)(jnp.asarray(D))
    Sig, Warp = np.array(Sig), np.array(Warp)
    I13 = np.trapezoid(Sig[:, :, 4], zg, axis=1)
    I23 = np.trapezoid(Sig[:, :, 3], zg, axis=1)
    q1, q2 = Q1.reshape(-1), Q2.reshape(-1)
    sc13 = np.where(np.abs(I13) > 1e-3 * np.abs(q1).max(), q1 / I13, 1.0)
    sc23 = np.where(np.abs(I23) > 1e-3 * np.abs(q2).max(), q2 / I23, 1.0)
    Sig[:, :, 4] *= sc13[:, None]
    Sig[:, :, 3] *= sc23[:, None]
    # sigma33 by the dynamic momentum integral
    wc = center_w_history(dat)
    kk = min(max(15 * (kd + 1) - 1, 1), len(wc) - 2)
    wdd = (wc[kk + 1] - 2 * wc[kk] + wc[kk - 1]) / DT ** 2
    S13g = Sig[:, :, 4].reshape(ngx, ngx, nz)
    S23g = Sig[:, :, 3].reshape(ngx, ngx, nz)
    d13 = np.gradient(S13g, xg, axis=1)
    d23 = np.gradient(S23g, xg, axis=0)
    rho = inp["material_db"][inp["mat_names"][0]]["rho"]
    acc = (s1 * s2) * wdd
    integrand = (rho * acc[:, :, None] - d13 - d23)
    s33 = np.concatenate(
        [np.zeros((ngx, ngx, 1)),
         np.cumsum(0.5 * (integrand[:, :, 1:] + integrand[:, :, :-1])
                   * np.diff(zg)[None, None, :], axis=2)], axis=2)
    Sig[:, :, 2] = s33.reshape(-1, nz)
    # 3-D displacement composition
    u0 = u0g.reshape(-1, 3)
    wxf, wyf = wx.reshape(-1), wy.reshape(-1)
    U1 = u0[:, 0:1] - zg[None, :] * wxf[:, None] + Warp[:, :, 0]
    U2 = u0[:, 1:2] - zg[None, :] * wyf[:, None] + Warp[:, :, 1]
    U3 = u0[:, 2:3] + Warp[:, :, 2]
    out = os.path.join(HERE, "ex2_rm_field.vtk")
    nn = ngx * ngx
    with open(out, "w") as f:
        f.write("# vtk DataFile Version 3.0\n"
                "Ex.2 OpenSG-RM dehomogenized field, step pulse, t ="
                " %.4f s (solid Gauss lattice)\nASCII\n"
                "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d %d\n"
                "POINTS %d float\n" % (tstar, ngx, ngx, nz, nn * nz))
        for iz in range(nz):
            for jj in range(ngx):
                for ii in range(ngx):
                    f.write("%.6e %.6e %.6e\n"
                            % (xg[ii], xg[jj], zg[iz] + H / 2))
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
    print("wrote %s (%dx%dx%d Gauss-lattice points)"
          % (os.path.basename(out), ngx, ngx, nz))
    for nm, c in (("S11", 0), ("S13", 4), ("S23", 3), ("S33", 2)):
        print("  max |%s| = %.4e Pa" % (nm, np.abs(Sig[:, :, c]).max()))


if __name__ == "__main__":
    main()
