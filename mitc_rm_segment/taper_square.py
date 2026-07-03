"""
taper_square.py -- linearly tapered SQUARE-tube segment, the flat-wall companion
to the circular taper study (taper_study.py).

Why a square: its walls are FLAT, so the shell initial (hoop) curvature is
k22 = 0 on every face and singular only at the four corners -- the opposite of
the circular tube (k22 = -1/R everywhere).  Comparing the two isolates how much
of the tapered bending--shear coupling error (C26, C35) is driven by the shell
initial curvature: it is the "use the shell initial curvature to find the
improvement" experiment.

Geometry: mid-surface square of half-width R(z) = R0 + (aR-1) R0 z/L, uniformly
scaled along the axis (so a1 = dP/dz tilts by the taper exactly as for the
circle), constant wall thickness t, CENTER reference (mid-surface mesh), corners
at nodes (NC a multiple of 4) so no element straddles a corner.

    python taper_square.py gen [nc nl]      shell + solid square meshes + red-ref PNGs
    python taper_square.py shell [mat]       general-RM shell L/taper/R for the sweep
    python taper_square.py tables [mat]      full 6x6 thin/thick tables (needs solid refs)

Solid runner: run_solid_study.py <mat> taper_square  (out/taper_square/{meshes,results}).
"""
import os, sys, math, json, time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import taper_study as ts

OUT = os.path.join(HERE, "out", "taper_square")
MESH = os.path.join(OUT, "meshes")
ORI = os.path.join(OUT, "orientation")
RES = os.path.join(OUT, "results")
for d in (MESH, ORI, RES):
    os.makedirs(d, exist_ok=True)

TAPERS = [1.0, 0.95, 0.9, 0.8, 0.7]
L, R0 = 2.0, 1.0
NC, NL, NR = 48, 10, 4           # NC multiple of 4


def _unit_square(nc):
    """nc perimeter points (CCW) of the half-width-1 square, corners on the grid;
    returns points Q (nc,2) and, per point, the inward normal of the edge that
    STARTS at that point (used for the per-element wall frame)."""
    ns = nc // 4
    Q, edge_in = [], []
    corners = [(1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)]
    # inward normal of each side (right, top, left, bottom)
    innor = [(-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0)]
    for s in range(4):
        a = np.array(corners[s]); b = np.array(corners[(s + 1) % 4])
        for m in range(ns):
            Q.append((a + (b - a) * m / ns).tolist())
            edge_in.append(innor[s])
    return np.array(Q), np.array(edge_in)


def gen_square_case(regime, mat, aR, mesh_dir=None, nc=None, nl=None, nr=None):
    mesh_dir = mesh_dir or MESH; os.makedirs(mesh_dir, exist_ok=True)
    nc = nc or NC; nl = nl or NL; nr = nr or NR
    t = ts.THICK[regime]
    matc = ts.ISO if mat == "iso" else ts.ANI
    layup = [["iso", t, 0.0]] if mat == "iso" else [["ani", t, -45.0]]
    tg = ts.tag_of(regime, mat, aR)
    Z = [L * i / nl for i in range(nl + 1)]
    dRdz = (aR * R0 - R0) / L
    Q, edge_in = _unit_square(nc)                      # (nc,2), (nc,2)

    def radius(z):
        return R0 + (aR * R0 - R0) * (z / L)

    # ---- shell (mid-surface square) ----
    snodes = []
    for z in Z:
        r = radius(z)
        for k in range(nc):
            snodes.append([float(r * Q[k, 0]), float(r * Q[k, 1]), float(z)])
    squads, soris = [], []
    for i in range(nl):
        for k in range(nc):
            k1 = (k + 1) % nc
            n0 = i * nc + k; n1 = i * nc + k1; n2 = (i + 1) * nc + k1; n3 = (i + 1) * nc + k
            squads.append([n0 + 1, n1 + 1, n2 + 1, n3 + 1])
            # element mid unit-square point + its wall inward normal (flat wall)
            qm = 0.5 * (Q[k] + Q[k1]); nin = edge_in[k]
            a2 = np.array([nin[1], -nin[0], 0.0])        # hoop tangent = rotate inward normal -90 (CCW)
            # ensure a2 points along increasing k (CCW): rotate inward +90 gives CCW tangent
            a2 = np.array([-nin[1], nin[0], 0.0])
            a1 = np.array([dRdz * qm[0], dRdz * qm[1], 1.0])   # generator (tilted by taper)
            a1 /= np.linalg.norm(a1)
            e3 = np.cross(a1, a2); e3 /= np.linalg.norm(e3)
            if e3[0] * nin[0] + e3[1] * nin[1] < 0:            # make e3 inward
                e3 = -e3
            e2 = np.cross(e3, a1); e2 /= np.linalg.norm(e2)
            soris.append(a1.tolist() + e2.tolist() + e3.tolist())
    shell = {"nodes": snodes, "elements": squads,
             "sections": [{"elementSet": "wall", "layup": layup}],
             "sets": {"element": [{"name": "wall", "labels": list(range(1, len(squads) + 1))}]},
             "materials": [{"name": matc["name"], "density": matc["rho"],
                            "elastic": {"E": matc["E"], "G": matc["G"], "nu": matc["nu"]}}],
             "elementOrientations": soris}
    yaml.safe_dump(shell, open(os.path.join(mesh_dir, "shell_%s.yaml" % tg), "w"),
                   default_flow_style=None, sort_keys=False)

    # ---- solid: miter-offset each mid node by +/- t/2 through NR layers ----
    # per-node inward miter vector m = (n1+n2)/(1+n1.n2) keeps both walls at t/2.
    node_off = np.zeros((nc, 2))
    for k in range(nc):
        n1 = edge_in[(k - 1) % nc]; n2 = edge_in[k]        # normals of the two edges meeting at node k
        n1 = np.array(n1); n2 = np.array(n2)
        node_off[k] = (n1 + n2) / (1.0 + float(n1 @ n2))

    def nid(i, mlay, k):
        return i * ((nr + 1) * nc) + mlay * nc + (k % nc) + 1
    dn = []
    for z in Z:
        r = radius(z)
        for mlay in range(nr + 1):
            off = (mlay / nr - 0.5) * t                     # -t/2 .. +t/2 (outward +)
            for k in range(nc):
                p = r * Q[k] - off * node_off[k]            # inward normal points to center; +off outward
                dn.append("%.10f %.10f %.10f" % (p[0], p[1], z))
    hexes, hories = [], []
    for i in range(nl):
        for mlay in range(nr):
            for k in range(nc):
                k1 = (k + 1) % nc
                hexes.append(" ".join(str(x) for x in
                                      [nid(i, mlay, k), nid(i, mlay, k1), nid(i, mlay + 1, k1), nid(i, mlay + 1, k),
                                       nid(i + 1, mlay, k), nid(i + 1, mlay, k1), nid(i + 1, mlay + 1, k1), nid(i + 1, mlay + 1, k)]))
                qm = 0.5 * (Q[k] + Q[k1]); nin = edge_in[k]
                a2 = np.array([-nin[1], nin[0], 0.0])
                a1 = np.array([dRdz * qm[0], dRdz * qm[1], 1.0]); a1 /= np.linalg.norm(a1)
                e3 = np.cross(a1, a2); e3 /= np.linalg.norm(e3)
                if e3[0] * nin[0] + e3[1] * nin[1] < 0:
                    e3 = -e3
                if mat == "iso":
                    e1 = a1
                else:
                    ca, sa = math.cos(math.radians(-45)), math.sin(math.radians(-45))
                    hoop = np.cross(e3, a1); hoop /= np.linalg.norm(hoop)
                    e1 = ca * a1 + sa * hoop
                e1 = e1 / np.linalg.norm(e1)
                e2 = np.cross(e3, e1)
                hories.append([float(v) for v in np.concatenate([e1, e2, e3])])
    mats = [{"name": matc["name"], "E": matc["E"], "G": matc["G"], "nu": matc["nu"], "rho": matc["rho"]}]
    solid = {"nodes": [[s] for s in dn], "elements": [[h] for h in hexes],
             "sets": {"element": [{"name": matc["name"], "labels": list(range(1, len(hexes) + 1))}]},
             "elementOrientations": hories, "materials": mats}
    yaml.safe_dump(solid, open(os.path.join(mesh_dir, "solid_%s.yaml" % tg), "w"),
                   default_flow_style=None, sort_keys=False)
    return tg


def red_ref_png(tg, mesh_dir=None, out_dir=None):
    """Solid tapered-square mesh with the mid-surface (center reference) drawn as a
    RED DOTTED loop at several axial stations -- shows the reference surface
    bisecting the wall thickness.  Returns the PNG path."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh_dir = mesh_dir or MESH; out_dir = out_dir or ORI
    os.makedirs(out_dir, exist_ok=True)
    snodes, selems, _ = ts._load_mesh(tg, "shell", mesh_dir)   # mid-surface nodes
    hn, he, _ = ts._load_mesh(tg, "solid", mesh_dir)
    cells = np.hstack([np.full((len(he), 1), 8), he]).ravel()
    ct = np.full(len(he), pv.CellType.HEXAHEDRON, np.uint8)
    grid = pv.UnstructuredGrid(cells, ct, hn)
    p = pv.Plotter(off_screen=True, window_size=(820, 940))
    p.add_mesh(grid, color="#c9c2b6", show_edges=True, edge_color="#8a8375", line_width=0.5, opacity=0.55)
    # red dotted mid-reference loops at each axial station present in the shell mesh
    zvals = np.unique(snodes[:, 2].round(6))
    for zi, z in enumerate(zvals):
        ring = snodes[np.isclose(snodes[:, 2], z)]
        if len(ring) < 4:
            continue
        loop = np.vstack([ring, ring[0]])
        pl = pv.MultipleLines(points=loop)
        p.add_mesh(pl, color="red", line_width=4, style="wireframe")
        # dotted look: also scatter the mid nodes big and red
        p.add_mesh(pv.PolyData(ring), color="red", point_size=6, render_points_as_spheres=True)
    p.add_text("SOLID square taper : %s\nred = mid-surface (center reference)" % tg, font_size=11)
    p.add_axes(line_width=3)
    p.camera_position = [(6.4, -6.4, 3.2), (0, 0, 1.0), (0, 0, 1)]
    fn = os.path.join(out_dir, "%s_solid_ref.png" % tg)
    p.screenshot(fn); p.close()
    return fn


def cmd_gen(nc=None, nl=None):
    for regime in ("thin", "thick"):
        for mat in ("iso", "m45"):
            for aR in TAPERS:
                tg = gen_square_case(regime, mat, aR, nc=nc, nl=nl)
                print("mesh", tg, "(NC=%d NL=%d)" % (nc or NC, nl or NL))
    for regime in ("thin", "thick"):
        for mat in ("iso", "m45"):
            red_ref_png(ts.tag_of(regime, mat, 0.7))
    print("red-ref PNGs ->", ORI)


def cmd_shell(mat):
    for regime in ("thin", "thick"):
        for aR in TAPERS:
            tg = ts.tag_of(regime, mat, aR)
            t0 = time.time()
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            print("shell %-18s  %6.1fs  taper diag(x1e9) %s"
                  % (tg, time.time() - t0, np.array2string(np.diag(S6) / 1e9, precision=3)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if cmd == "gen":
        cmd_gen(int(sys.argv[2]) if len(sys.argv) > 2 else None,
                int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == "shell":
        cmd_shell(sys.argv[2] if len(sys.argv) > 2 else "iso")
