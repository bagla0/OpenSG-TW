"""rpt_to_vtk_ex2.py -- Ex.2: convert the Abaqus Gauss-point field report
of the 3-D benchmark (ex2_solid_S.rpt / ex2_solid_U.rpt, written by
odb_rpt_ex2.py at the matched 0.75 ms peak frame) into a VTK volume on the
40x40x36 Gauss lattice, for the identical-pipeline ParaView comparison
against the OpenSG-RM field.

Same conventions as the ex5 converter it is ported from:
  * the S rpt is GLOBAL-frame (viewport transformation inherited) with 16
    columns (labels + invariants first, the six components last);
  * the U rpt has a Magnitude column before U1/U2/U3 (5 columns);
  * everything is sign-flipped to the shell's +z load convention;
  * the ex2 solid mesh is UNIFORM through the thickness (18 layers of
    h/18: 6 elements per ply of the (0/90/0) stack), so the Gauss depths
    are the uniform-lattice ones.

Run:  python examples/opensg-rm_dynamic/ex2/rpt_to_vtk_ex2.py
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NX, NZT = 20, 18
A, H = 0.762, 0.1524
DX, DZ = A / NX, H / NZT
G = 0.5 / np.sqrt(3.0)


def read_rpt(path, ncols):
    rows = []
    with open(path, errors="replace") as f:
        for ln in f:
            toks = ln.split()
            if not toks or not re.fullmatch(r"\d+", toks[0]):
                continue
            vals = []
            ok = True
            for t in toks:
                try:
                    vals.append(float(t))
                except ValueError:
                    ok = False
                    break
            if ok and len(vals) >= ncols:
                rows.append(vals[:ncols])
    return np.array(rows)


def main():
    d = os.path.join(HERE, "Abaqus_results")
    S = read_rpt(os.path.join(d, "ex2_solid_S.rpt"), 16)
    U = read_rpt(os.path.join(d, "ex2_solid_U.rpt"), 5)
    print("parsed %d Gauss rows, %d node rows" % (len(S), len(U)))
    ngx, ngz = 2 * NX, 2 * NZT
    SG = np.zeros((ngz, ngx, ngx, 6))
    for row in S:
        e = int(row[0]) - 1
        ipt = int(row[1]) - 1
        s = -row[10:16]                        # global; -z deck -> +z
        k, j, i = e // (NX * NX), (e % (NX * NX)) // NX, e % NX
        gx = 2 * i + (ipt % 2)
        gy = 2 * j + ((ipt // 2) % 2)
        gz = 2 * k + (ipt // 4)
        SG[gz, gy, gx] = s
    npl = (NX + 1) * (NX + 1)
    UN = np.zeros((NZT + 1, NX + 1, NX + 1, 3))
    for row in U:
        n = int(row[0]) - 1
        k, j, i = n // npl, (n % npl) // (NX + 1), n % (NX + 1)
        UN[k, j, i] = -row[2:5]
    xg = (np.repeat(np.arange(NX), 2) + 0.5 + np.tile([-G, +G], NX)) * DX
    zg = (np.repeat(np.arange(NZT), 2) + 0.5 + np.tile([-G, +G], NZT)) * DZ

    def trilerp(P):
        out = np.empty((ngz, ngx, ngx, 3))
        fx = xg / DX
        i0 = np.clip(fx.astype(int), 0, NX - 1)
        tx = fx - i0
        k0 = np.repeat(np.arange(NZT), 2)
        tz = np.tile([0.5 - G, 0.5 + G], NZT)
        for kz in range(ngz):
            k, wz = k0[kz], tz[kz]
            Pz = (1 - wz) * P[k] + wz * P[k + 1]
            for jy in range(ngx):
                jj, wy = i0[jy], tx[jy]
                Py = (1 - wy) * Pz[jj] + wy * Pz[jj + 1]
                out[kz, jy] = ((1 - tx)[:, None] * Py[i0]
                               + tx[:, None] * Py[i0 + 1])
        return out

    UGL = trilerp(UN)
    out = os.path.join(HERE, "ex2_solid_field.vtk")
    npts = ngx * ngx * ngz
    with open(out, "w") as f:
        f.write("# vtk DataFile Version 3.0\n"
                "Ex.2 Abaqus 3-D solid field from the Gauss rpt (global,"
                " +z convention, t = 0.75 ms)\nASCII\n"
                "DATASET STRUCTURED_GRID\nDIMENSIONS %d %d %d\n"
                "POINTS %d float\n" % (ngx, ngx, ngz, npts))
        for kz in range(ngz):
            for jy in range(ngx):
                for ix in range(ngx):
                    f.write("%.6e %.6e %.6e\n" % (xg[ix], xg[jy], zg[kz]))
        f.write("POINT_DATA %d\nVECTORS disp float\n" % npts)
        for kz in range(ngz):
            for jy in range(ngx):
                for ix in range(ngx):
                    u = UGL[kz, jy, ix]
                    f.write("%.6e %.6e %.6e\n" % (u[0], u[1], u[2]))
        for c, nm in enumerate(("S11", "S22", "S33", "S12", "S13", "S23")):
            f.write("SCALARS %s float 1\nLOOKUP_TABLE default\n" % nm)
            for kz in range(ngz):
                for jy in range(ngx):
                    for ix in range(ngx):
                        f.write("%.6e\n" % SG[kz, jy, ix, c])
    print("wrote %s (%dx%dx%d Gauss points)"
          % (os.path.basename(out), ngx, ngx, ngz))
    for nm, c in (("S11", 0), ("S13", 4), ("S23", 5), ("S33", 2)):
        print("  max |%s| = %.4e Pa" % (nm, np.abs(SG[:, :, :, c]).max()))


if __name__ == "__main__":
    main()
