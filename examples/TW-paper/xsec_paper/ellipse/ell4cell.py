"""ell4cell.py -- PRISMATIC elliptic 4-CELL tube (3 internal webs) cross-section, center
reference, for the RM cross-section paper.  Geometry adapted from the validated tapered
run_ell3w.py (webs at x = cos(60/90/120 deg)*a = +0.5,0,-0.5), made prismatic (constant
a=1.0,b=0.6, wall t=0.02) and 2-D.

Emits, for material in {iso, m45}:
  shell_ell4cell_<mat>.yaml : 1-D RM shell contour (skin + web lines; e1=beam, e2=tangent,
                              e3=inward normal; layup carries the ply angle) -> ring_indep 6-DOF
  solid_ell4cell_<mat>.yaml : 2-D solid quad mesh (nr through the wall, webs watertight at the
                              skin junctions; fiber baked into e1) -> compute_timo_from_yaml

  python ell4cell.py gen [nc]      # write meshes (default nc=120)
  python ell4cell.py run [nc]      # gen + homogenize (RM ring vs 2-D solid) full 6x6 + %err
  python ell4cell.py conv          # mesh convergence (nc sweep)
"""
import os
import sys
import math

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
XSEC = os.path.abspath(os.path.join(HERE, ".."))
for q in (MITC, REPO, XSEC):
    sys.path.insert(0, q)

A, B, T, NR = 1.0, 0.6, 0.02, 4
MATS = {
    "iso": dict(name="iso", rho=1800.0, E=[70e9, 70e9, 70e9], G=[26.923e9] * 3, nu=[0.3] * 3, ply=0.0),
    "m45": dict(name="ani", rho=1800.0, E=[37e9, 9e9, 9e9], G=[4e9] * 3, nu=[0.3] * 3, ply=-45.0),
}
OUT = os.path.join(HERE, "meshes")
os.makedirs(OUT, exist_ok=True)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


class _Flow(list):
    pass


yaml.add_representer(_Flow, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))


def _skin_pt(k, nc):
    th = 2 * math.pi * k / nc
    return np.array([A * math.cos(th), B * math.sin(th)]), th


def _skin_frame(th):
    """prismatic skin wall frame: tangent (hoop), inward normal."""
    tan = np.array([-A * math.sin(th), B * math.cos(th)]); tan /= np.linalg.norm(tan)
    n_out = np.array([B * math.cos(th), A * math.sin(th)]); n_out /= np.linalg.norm(n_out)
    return tan, n_out                                       # e2 = tan ; outward normal


def gen(nc, nw=None):
    nw = nw or max(6, nc // 10)
    ktops = [nc // 6, nc // 4, nc // 3]
    csw = [math.cos(2 * math.pi * k / nc) for k in ktops]
    for mkey, M in MATS.items():
        cpl, spl = math.cos(math.radians(M["ply"])), math.sin(math.radians(M["ply"]))
        # ---------------- 1-D SHELL CONTOUR ----------------
        snodes = [[float(_skin_pt(k, nc)[0][0]), float(_skin_pt(k, nc)[0][1]), 0.0] for k in range(nc)]
        selems, soris = [], []
        for k in range(nc):                                # skin line elements
            k1 = (k + 1) % nc
            selems.append([k + 1, k1 + 1])
            _, th = _skin_pt(k + 0.5 if k1 else 0.5, nc)
            tan, n_out = _skin_frame(2 * math.pi * (k + 0.5) / nc)
            e2 = [float(tan[0]), float(tan[1]), 0.0]; e3 = [float(-n_out[0]), float(-n_out[1]), 0.0]
            soris.append([0.0, 0.0, 1.0] + e2 + e3)
        web_node0 = nc
        wnode = {}
        for w, c in enumerate(csw):
            s = math.sqrt(1 - c * c)
            for j in range(1, nw):
                eta = -1.0 + 2.0 * j / nw
                wnode[(w, j)] = len(snodes)
                snodes.append([c * A, float(eta * s * B), 0.0])

        def scid(w, j):                                    # shell contour web node id (0-based)
            if j == 0:
                return nc - ktops[w]
            if j == nw:
                return ktops[w]
            return wnode[(w, j)]
        for w, c in enumerate(csw):                        # web line elements
            for j in range(nw):
                selems.append([scid(w, j) + 1, scid(w, j + 1) + 1])
                soris.append([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])  # e2=+y, e3=+x
        shell = {
            "nodes": [_Flow(n) for n in snodes],
            "elements": [_Flow(e) for e in selems],
            "sections": [{"type": "shell", "elementSet": "wall",
                          "layup": [["mat", float(T), float(M["ply"])]]}],
            "sets": {"element": [{"name": "wall", "labels": _Flow(list(range(1, len(selems) + 1)))}]},
            "materials": [{"name": "mat", "density": M["rho"],
                           "elastic": {"E": _Flow(M["E"]), "G": _Flow(M["G"]), "nu": _Flow(M["nu"])}}],
            "elementOrientations": [_Flow(o) for o in soris],
        }
        yaml.dump(shell, open(os.path.join(OUT, "shell_ell4cell_%s.yaml" % mkey), "w"),
                  default_flow_style=None, sort_keys=False)

        # ---------------- 2-D SOLID (quad, watertight webs) ----------------
        dn = []
        def sid(m, k):
            return m * nc + (k % nc)
        for m in range(NR + 1):
            off = (m / NR - 0.5) * T
            for k in range(nc):
                p, th = _skin_pt(k, nc)
                _, n_out = _skin_frame(th)
                q = p + off * n_out
                dn.append([float(q[0]), float(q[1]), 0.0])
        wbase = len(dn)
        for w, c in enumerate(csw):
            s = math.sqrt(1 - c * c)
            for j in range(1, nw):
                eta = -1.0 + 2.0 * j / nw
                for m in range(NR + 1):
                    off = (m / NR - 0.5) * T
                    dn.append([float(c * A + off), float(eta * s * B), 0.0])

        def wsid(w, j, m):
            if j == 0:
                return sid(m, nc - ktops[w])
            if j == nw:
                return sid(m, ktops[w])
            return wbase + w * (nw - 1) * (NR + 1) + (j - 1) * (NR + 1) + m
        quads, qori = [], []

        def wallframe_solid(th):
            tan, n_out = _skin_frame(th)
            a1 = np.array([0.0, 0.0, 1.0]); a2 = np.array([tan[0], tan[1], 0.0])
            e3 = np.array([-n_out[0], -n_out[1], 0.0])      # inward
            e1 = cpl * a1 + spl * a2; e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)
            return np.concatenate([e1, e2, e3]).tolist()
        for m in range(NR):                                # skin quads
            for k in range(nc):
                k1 = (k + 1) % nc
                quads.append([sid(m, k) + 1, sid(m, k1) + 1, sid(m + 1, k1) + 1, sid(m + 1, k) + 1])
                qori.append(wallframe_solid(2 * math.pi * (k + 0.5) / nc))
        for w, c in enumerate(csw):                        # web quads (through-thickness along x)
            a1 = np.array([0.0, 0.0, 1.0]); a2 = np.array([0.0, 1.0, 0.0]); e3 = np.array([1.0, 0.0, 0.0])
            e1 = cpl * a1 + spl * a2; e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
            wo = np.concatenate([e1, e2, e3]).tolist()
            for j in range(nw):
                for m in range(NR):
                    quads.append([wsid(w, j, m) + 1, wsid(w, j + 1, m) + 1,
                                  wsid(w, j + 1, m + 1) + 1, wsid(w, j, m + 1) + 1])
                    qori.append(wo)
        solid = {
            "nodes": [_Flow(["%.10f %.10f %.10f" % tuple(p)]) for p in dn],
            "elements": [_Flow([" ".join(str(x) for x in q)]) for q in quads],
            "sets": {"element": [{"name": M["name"], "labels": _Flow(list(range(1, len(quads) + 1)))}]},
            "elementOrientations": [_Flow(o) for o in qori],
            "materials": [{"name": M["name"], "E": _Flow(M["E"]), "G": _Flow(M["G"]),
                           "nu": _Flow(M["nu"]), "rho": M["rho"]}],
        }
        yaml.dump(solid, open(os.path.join(OUT, "solid_ell4cell_%s.yaml" % mkey), "w"),
                  default_flow_style=None, sort_keys=False)
        print("  gen %s nc=%d nw=%d: shell %d nodes/%d elems | solid %d nodes/%d quads"
              % (mkey, nc, nw, len(snodes), len(selems), len(dn), len(quads)), flush=True)
    return dict(nc=nc, nw=nw, ktops=ktops)


def homogenize(mkey):
    from xsec_5v6_master import load_ring, ring_6dof
    from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
    sym = lambda X: 0.5 * (np.asarray(X) + np.asarray(X).T)
    ring = ring_6dof(load_ring(os.path.join(OUT, "shell_ell4cell_%s.yaml" % mkey)))
    solid = sym(compute_timo_from_yaml(os.path.join(OUT, "solid_ell4cell_%s.yaml" % mkey), verbose=False))
    return solid, ring


def _show(mkey, solid, ring):
    print("\n#### elliptic 4-cell  %s  (solid | RM 6-DOF | %%err, diagonal) ####" % mkey)
    for i in range(6):
        e = 100 * (ring[i, i] - solid[i, i]) / solid[i, i]
        print("  %-4s solid=% .4e  RM=% .4e  err=%+6.2f%%" % (LBL[i], solid[i, i], ring[i, i], e))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "gen":
        gen(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
    elif cmd == "run":
        nc = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        gen(nc)
        res = {}
        for mkey in ("iso", "m45"):
            solid, ring = homogenize(mkey)
            _show(mkey, solid, ring)
            res[mkey] = dict(solid=solid, ring=ring)
        np.savez(os.path.join(HERE, "ell4cell_%d.npz" % nc),
                 **{"%s_solid" % k: v["solid"] for k, v in res.items()},
                 **{"%s_ring" % k: v["ring"] for k, v in res.items()})
    elif cmd == "conv":
        rows = []
        for nc in (48, 72, 96, 144, 192, 288):
            gen(nc)
            line = [nc]
            for mkey in ("iso", "m45"):
                solid, ring = homogenize(mkey)
                line.append([100 * (ring[i, i] - solid[i, i]) / solid[i, i] for i in range(6)])
            rows.append(line)
            print("nc=%-4d iso %s | m45 %s"
                  % (nc, " ".join("%+5.2f" % v for v in line[1]), " ".join("%+5.2f" % v for v in line[2])), flush=True)
        np.savez(os.path.join(HERE, "ell4cell_conv.npz"),
                 nc=np.array([r[0] for r in rows]),
                 iso=np.array([r[1] for r in rows]), m45=np.array([r[2] for r in rows]), labels=LBL)
