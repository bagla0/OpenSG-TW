"""run_ell3w.py -- TAPERED ELLIPSE + 3 WEBS (blade-like multi-cell), m45, center ref.

Skin: a(z): 1.0->0.65, b(z): 0.6->0.42 over L=2.0, t=0.02, [-45] ply.
Webs: vertical planes x = c*a(z), c = cos(60/90/120 deg) = +0.5, 0, -0.5 --
      tapered with the section, meeting the skin at parametric stations that are
      mesh nodes at every refinement (k_top = nc/6, nc/4, nc/3).
Shell: branched mid-surface mesh (skin quads + web quads, C0-shared junction lines).
Solid: conforming hexes -- web end columns REUSE the skin's through-thickness node
       columns at the junctions (watertight T-junction).
Compare 6-DOF shell (full / flux-tied / canonical MITC) vs fresh 3-D solid.
"""
import os, sys, math, time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)

L, A0, A1, B0, B1, T = 2.0, 1.0, 0.65, 0.6, 0.42, 0.02
ANI = dict(name="ani", rho=1800.0, E=[37e9, 9e9, 9e9], G=[4e9] * 3, nu=[0.3] * 3)
CA, SA = math.cos(math.radians(-45)), math.sin(math.radians(-45))
OUT = os.path.join(HERE, "out", "ell3w")
os.makedirs(OUT, exist_ok=True)


def ab(z):
    return A0 + (A1 - A0) * z / L, B0 + (B1 - B0) * z / L


def skin_frame(th, z):
    a, b = ab(z)
    da, db = (A1 - A0) / L, (B1 - B0) / L
    gen = np.array([da * math.cos(th), db * math.sin(th), 1.0])
    hoop = np.array([-a * math.sin(th), b * math.cos(th), 0.0])
    a2 = hoop / np.linalg.norm(hoop)
    a1 = gen - (gen @ a2) * a2; a1 /= np.linalg.norm(a1)
    e3 = np.cross(a1, a2); e3 /= np.linalg.norm(e3)
    n2d = np.array([b * math.cos(th), a * math.sin(th), 0.0]); n2d /= np.linalg.norm(n2d)
    if e3 @ (-n2d) < 0:
        e3 = -e3; a1 = -a1
    return a1, a2, e3, n2d


def web_frame(c, eta, z):
    """web plane x = c*a(z); eta in [-1,1] along the web (bottom->top)."""
    a, b = ab(z)
    da, db = (A1 - A0) / L, (B1 - B0) / L
    s = math.sqrt(1.0 - c * c)
    gen = np.array([c * da, eta * s * db, 1.0])
    e2 = np.array([0.0, 1.0, 0.0])
    a1 = gen - (gen @ e2) * e2; a1 /= np.linalg.norm(a1)
    e3 = np.cross(a1, e2); e3 /= np.linalg.norm(e3)
    if e3[0] < 0:
        e3 = -e3; a1 = -a1                      # normal toward +x
    return a1, e2, e3


def gen_ell3w(mesh_dir, nc, nl, nw, nr=4):
    """Branched shell + conforming solid.  Webs at k_top = nc/6, nc/4, nc/3."""
    os.makedirs(mesh_dir, exist_ok=True)
    Z = [L * i / nl for i in range(nl + 1)]
    ktops = [nc // 6, nc // 4, nc // 3]
    csw = [math.cos(2 * math.pi * k / nc) for k in ktops]

    # ---------------- SHELL ----------------
    snodes, squads, soris = [], [], []
    for z in Z:
        a, b = ab(z)
        for k in range(nc):
            th = 2 * math.pi * k / nc
            snodes.append([a * math.cos(th), b * math.sin(th), float(z)])
    nweb_int = nw - 1                                    # interior nodes per web per ring
    web_base = len(Z) * nc
    for iz, z in enumerate(Z):
        a, b = ab(z)
        for w, c in enumerate(csw):
            s = math.sqrt(1 - c * c)
            for j in range(1, nw):                       # eta from -1 (bottom) to +1
                eta = -1.0 + 2.0 * j / nw
                snodes.append([c * a, eta * s * b, float(z)])

    def wid(iz, w, j):
        """global node id (0-based) of web w, interior column j (1..nw-1), ring iz;
        j=0 -> bottom skin node, j=nw -> top skin node."""
        if j == 0:
            return iz * nc + (nc - ktops[w])             # bottom station (theta = -th_w)
        if j == nw:
            return iz * nc + ktops[w]
        return web_base + iz * (3 * nweb_int) + w * nweb_int + (j - 1)

    for i in range(nl):
        zm = (Z[i] + Z[i + 1]) / 2
        for k in range(nc):                              # skin quads
            k1 = (k + 1) % nc
            squads.append([i * nc + k + 1, i * nc + k1 + 1,
                           (i + 1) * nc + k1 + 1, (i + 1) * nc + k + 1])
            a1, a2, e3, _ = skin_frame(2 * math.pi * (k + 0.5) / nc, zm)
            soris.append(a1.tolist() + a2.tolist() + e3.tolist())
        for w, c in enumerate(csw):                      # web quads
            for j in range(nw):
                n0 = wid(i, w, j); n1 = wid(i, w, j + 1)
                n2 = wid(i + 1, w, j + 1); n3 = wid(i + 1, w, j)
                squads.append([n0 + 1, n1 + 1, n2 + 1, n3 + 1])
                a1, e2, e3 = web_frame(c, -1.0 + (2.0 * j + 1.0) / nw, zm)
                soris.append(a1.tolist() + e2.tolist() + e3.tolist())
    shell = {"nodes": snodes, "elements": squads,
             "sections": [{"elementSet": "wall", "layup": [["ani", T, -45.0]]}],
             "sets": {"element": [{"name": "wall", "labels": list(range(1, len(squads) + 1))}]},
             "materials": [{"name": ANI["name"], "density": ANI["rho"],
                            "elastic": {"E": ANI["E"], "G": ANI["G"], "nu": ANI["nu"]}}],
             "elementOrientations": soris}
    yaml.safe_dump(shell, open(os.path.join(mesh_dir, "shell_e3w.yaml"), "w"),
                   default_flow_style=None, sort_keys=False)

    # ---------------- SOLID (conforming) ----------------
    # skin sheets: (nr+1) offsets along n2d; web sheets: (nr+1) offsets along +x,
    # END columns of each web reuse the skin's through-thickness column at the station.
    dn = []
    def sid(iz, m, k):
        return iz * ((nr + 1) * nc) + m * nc + (k % nc)
    for z in Z:
        a, b = ab(z)
        for m in range(nr + 1):
            off = (m / nr - 0.5) * T
            for k in range(nc):
                th = 2 * math.pi * k / nc
                _, _, _, n2d = skin_frame(th, z)
                p = np.array([a * math.cos(th), b * math.sin(th)]) + off * n2d[:2]
                dn.append([p[0], p[1], z])
    wsolid_base = len(dn)
    for iz, z in enumerate(Z):
        a, b = ab(z)
        for w, c in enumerate(csw):
            s = math.sqrt(1 - c * c)
            for j in range(1, nw):
                eta = -1.0 + 2.0 * j / nw
                for m in range(nr + 1):
                    off = (m / nr - 0.5) * T
                    dn.append([c * a + off, eta * s * b, z])

    def wsid(iz, w, j, m):
        if j == 0:
            return sid(iz, m, nc - ktops[w])
        if j == nw:
            return sid(iz, m, ktops[w])
        return (wsolid_base + iz * (3 * nweb_int * (nr + 1))
                + w * (nweb_int * (nr + 1)) + (j - 1) * (nr + 1) + m)

    hexes, hories = [], []
    for i in range(nl):
        zm = (Z[i] + Z[i + 1]) / 2
        for m in range(nr):
            for k in range(nc):                          # skin hexes
                k1 = (k + 1) % nc
                hexes.append([sid(i, m, k), sid(i, m, k1), sid(i, m + 1, k1), sid(i, m + 1, k),
                              sid(i + 1, m, k), sid(i + 1, m, k1), sid(i + 1, m + 1, k1), sid(i + 1, m + 1, k)])
                a1, a2, e3, _ = skin_frame(2 * math.pi * (k + 0.5) / nc, zm)
                e1 = CA * a1 + SA * a2; e1 /= np.linalg.norm(e1)
                hories.append([float(v) for v in np.concatenate([e1, np.cross(e3, e1), e3])])
        for w, c in enumerate(csw):                      # web hexes
            for j in range(nw):
                for m in range(nr):
                    hexes.append([wsid(i, w, j, m), wsid(i, w, j + 1, m),
                                  wsid(i, w, j + 1, m + 1), wsid(i, w, j, m + 1),
                                  wsid(i + 1, w, j, m), wsid(i + 1, w, j + 1, m),
                                  wsid(i + 1, w, j + 1, m + 1), wsid(i + 1, w, j, m + 1)])
                    a1, e2, e3 = web_frame(c, -1.0 + (2.0 * j + 1.0) / nw, zm)
                    e1 = CA * a1 + SA * e2; e1 /= np.linalg.norm(e1)
                    hories.append([float(v) for v in np.concatenate([e1, np.cross(e3, e1), e3])])
    solid = {"nodes": [["%.10f %.10f %.10f" % tuple(p)] for p in dn],
             "elements": [[" ".join(str(x + 1) for x in h)] for h in hexes],
             "sets": {"element": [{"name": ANI["name"], "labels": list(range(1, len(hexes) + 1))}]},
             "elementOrientations": hories,
             "materials": [{"name": ANI["name"], "E": ANI["E"], "G": ANI["G"],
                            "nu": ANI["nu"], "rho": ANI["rho"]}]}
    yaml.safe_dump(solid, open(os.path.join(mesh_dir, "solid_e3w.yaml"), "w"),
                   default_flow_style=None, sort_keys=False)
    print("gen %s: shell %d nodes %d quads | solid %d nodes %d hexes"
          % (mesh_dir, len(snodes), len(squads), len(dn), len(hexes)))


def solid_ref():
    npz = os.path.join(OUT, "solid_ref.npz")
    if os.path.exists(npz):
        print("solid ref exists"); return np.load(npz)
    sys.path.insert(0, os.path.expanduser("~/claude_tmp/opensg-FEniCS"))
    from opensg.mesh.segment import SolidSegmentMesh
    from opensg.core.solid import compute_stiffness
    mdir = os.path.join(OUT, "solid_mesh")
    gen_ell3w(mdir, nc=96, nl=20, nw=12, nr=4)
    t0 = time.time()
    sm = SolidSegmentMesh(os.path.join(mdir, "solid_e3w.yaml"))
    mp, dens = sm.material_database
    bnd = compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=False)[0]
    tap = np.asarray(compute_stiffness(mp, sm.meshdata, sm.left_submesh, sm.right_submesh, Taper=True)[0])
    print("solid done %.0fs  EA(seg)=%.4e" % (time.time() - t0, tap[0, 0]))
    np.savez(npz, L=np.asarray(bnd[0]), R=np.asarray(bnd[1]), seg=tap)
    return np.load(npz)


def main():
    ref = solid_ref()
    So = 0.5 * (ref["seg"] + ref["seg"].T)
    import run_indep as ri
    import segment_indep
    orig = segment_indep.assemble_segment_indep
    LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
    mdir = os.path.join(OUT, "shell_48x10")
    gen_ell3w(mdir, nc=48, nl=10, nw=6, nr=4)
    print("\nELLIPSE + 3 TAPERED WEBS, m45 thin: segment %err vs 3-D solid (96x20,nw12,x4)")
    print("%-24s" % "scheme" + "".join("%8s" % l for l in LBL))
    for sch, name in (("full", "no MITC (full int.)"),
                      ("mitc4_wonly", "flux-tied"),
                      ("mitc4_both", "canonical MITC")):
        def patched(*a, **k):
            k["shear"] = sch
            return orig(*a, **k)
        segment_indep.assemble_segment_indep = patched
        import importlib
        importlib.reload(ri)
        S = ri.shell_solve_lagrange("e3w", mdir, os.path.join(OUT, "res"))
        e = [100 * (S[i, i] - So[i, i]) / So[i, i] for i in range(6)]
        print("%-24s" % name + "".join("%+7.1f " % v for v in e))
        sys.stdout.flush()
    segment_indep.assemble_segment_indep = orig
    print("\nsolid seg diag (x1e9):", np.round(np.diag(So) / 1e9, 4))


if __name__ == "__main__":
    main()
