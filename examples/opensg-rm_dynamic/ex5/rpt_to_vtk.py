"""rpt_to_vtk.py -- convert the Abaqus field REPORT (the raw .rpt over ALL
Gauss/integration points) into a VTK volume, so the Abaqus benchmark field
and the OpenSG-RM field (rm_field_vtk.py) are contoured through the SAME
ParaView pipeline.

Inputs (written by odb_rpt.py on the Abaqus machine, copied into
Abaqus_results/):
    sandwich_solid_step_S.rpt   S11..S23 at every integration point
                                (6400 C3D8I x 8 Gauss points = 51,200 rows)
    sandwich_solid_step_U.rpt   U1..U3 at every node (7,497 rows)

What this converter does, transparently:
  1. parse the rpt tables (element label, int-pt number, six stresses /
     node label, three displacements);
  2. place every Gauss point on the regular 40x40x32 Gauss lattice of the
     20x20x16 box mesh: C3D8 integration points are numbered xi-fastest
     (1:(-,-,-), 2:(+,-,-), 3:(-,+,-), 4:(+,+,-), 5..8 the +zeta four),
     at +-1/sqrt(3) of the element natural coordinates;
  3. the rpt is ALREADY in the global frame ("Coordinate system: GLOB" in
     its header -- writeFieldReport inherits the viewport transformation
     set by odb_rpt.py), so NO ply rotation is applied here; the component
     columns are the last six of the 16-column table (the first ten are
     labels + the stress invariants Mises/principals/Tresca/pressure);
  4. flip the sign of everything (the solid deck loads -z while the RM
     shell loads +z -- linear response, global flip) so both vtk files
     live in the same convention;
  5. interpolate the nodal U trilinearly to the Gauss lattice;
  6. write sandwich_solid_field.vtk: STRUCTURED_GRID 40x40x32 with the
     disp vector + all six stresses as POINT data -- ALL Gauss values
     enter, none are averaged away.

Run:  python examples/opensg-rm_dynamic/ex5/rpt_to_vtk.py
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NX, NZT = 20, 16
A, H = 1.524, 0.1524
DX = A / NX
G = 0.5 / np.sqrt(3.0)          # Gauss offset in HALF-element units
# the solid mesh is PLY-RESOLVED, not uniform: 4 x 1.905 mm plies, 8 core
# sub-layers of 17.145 mm, 4 x 1.905 mm plies (bottom -> top)
TPLY, TCORE = 0.0125 * H, 0.9 * H
TLAY = [TPLY] * 4 + [TCORE / 8.0] * 8 + [TPLY] * 4
ZK = np.concatenate([[0.0], np.cumsum(TLAY)])   # the 17 node planes
# bottom->top layer angles of the (0/90/0/90/core/90/0/90/0) stack
ANGS = [0.0, 90.0, 0.0, 90.0] + [0.0] * 8 + [90.0, 0.0, 90.0, 0.0]


def read_rpt(path, ncols):
    """All data rows of an Abaqus field report: rows whose first token is
    an integer label; returns the numeric matrix."""
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
    # 16 columns: El, IntPt, Mises, MaxPrin, MaxPrin(a), MidPrin, MinPrin,
    # Tresca, Pressure, ThirdInv, S11, S22, S33, S12, S13, S23 (GLOBAL)
    S = read_rpt(os.path.join(d, "sandwich_solid_step_S.rpt"), 16)
    U = read_rpt(os.path.join(d, "sandwich_solid_step_U.rpt"), 4)
    print("parsed %d Gauss rows, %d node rows" % (len(S), len(U)))
    # ---- Gauss lattice (2 per element per direction) --------------------
    ngx, ngz = 2 * NX, 2 * NZT
    SG = np.zeros((ngz, ngx, ngx, 6))          # [kz, jy, ix, comp]
    for row in S:
        e = int(row[0]) - 1
        ipt = int(row[1]) - 1
        s = -row[10:16]                        # global; -z deck -> +z
        k, j, i = e // (NX * NX), (e % (NX * NX)) // NX, e % NX
        gx = 2 * i + (ipt % 2)
        gy = 2 * j + ((ipt // 2) % 2)
        gz = 2 * k + (ipt // 4)
        SG[gz, gy, gx] = s
    # ---- nodal displacements -> trilinear at the Gauss lattice ---------
    npl = (NX + 1) * (NX + 1)
    UN = np.zeros((NZT + 1, NX + 1, NX + 1, 3))
    for row in U:
        n = int(row[0]) - 1
        k, j, i = n // npl, (n % npl) // (NX + 1), n % (NX + 1)
        UN[k, j, i] = -row[1:4]                # same sign flip
    # physical Gauss coordinates: in-plane uniform, THROUGH-THICKNESS at
    # the true ply-resolved layer Gauss depths
    xg = (np.repeat(np.arange(NX), 2) + 0.5
          + np.tile([-G, +G], NX) * 1.0) * DX
    zg = np.empty(2 * NZT)
    for k in range(NZT):
        zg[2 * k] = ZK[k] + (0.5 - G) * TLAY[k]
        zg[2 * k + 1] = ZK[k] + (0.5 + G) * TLAY[k]

    def trilerp(P):
        """nodal grid (NZT+1, NX+1, NX+1, 3) -> Gauss lattice values."""
        out = np.empty((ngz, ngx, ngx, 3))
        fx = xg / DX
        i0 = np.clip(fx.astype(int), 0, NX - 1)
        tx = fx - i0
        k0 = np.repeat(np.arange(NZT), 2)          # layer of each station
        tz = np.tile([0.5 - G, 0.5 + G], NZT)      # weight inside layer
        for kz in range(ngz):
            k, wz = k0[kz], tz[kz]
            Pz = (1 - wz) * P[k] + wz * P[k + 1]      # (NX+1, NX+1, 3)
            for jy in range(ngx):
                jj, wy = i0[jy], tx[jy]
                Py = (1 - wy) * Pz[jj] + wy * Pz[jj + 1]
                out[kz, jy] = ((1 - tx)[:, None] * Py[i0]
                               + tx[:, None] * Py[i0 + 1])
        return out

    UGL = trilerp(UN)
    # ---- write the VTK --------------------------------------------------
    out = os.path.join(HERE, "sandwich_solid_field.vtk")
    npts = ngx * ngx * ngz
    with open(out, "w") as f:
        f.write("# vtk DataFile Version 3.0\n"
                "Abaqus 3-D solid field from the Gauss-point rpt (global"
                " frame, +z convention)\nASCII\nDATASET STRUCTURED_GRID\n"
                "DIMENSIONS %d %d %d\nPOINTS %d float\n"
                % (ngx, ngx, ngz, npts))
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
    for c, nm in enumerate(("S11", "S13", "S23", "S33")):
        idx = {"S11": 0, "S13": 4, "S23": 5, "S33": 2}[nm]
        print("  max |%s| = %.4e Pa" % (nm, np.abs(SG[:, :, :, idx]).max()))


if __name__ == "__main__":
    main()
