"""
FEniCS 2D-solid Timoshenko 6x6 for flat plate-strips (rectangle W x H), run in
WSL (opensg_env_v8).  Isotropic and anisotropic [45/-45] strips, W=1 m, over 5
thickness ratios h/W.  The [45/-45] stack is two through-thickness bands
(+45 top, -45 bottom), fiber baked into the element frame EE1.  Saves all 6x6 to
data/strip_solid_6x6.csv.  Order [ext, shear2, shear3, twist, bend2, bend3].

Rectangle is centred at the origin (y2 in [-W/2,W/2], y3 in [-H/2,H/2]) so its
centroid coincides with the centre-referenced shell.  Material frame per element:
EE1 = fiber = sin(th) y2-tangent + cos(th) beam, EE3 = +y3 (through-thickness).
"""
import os, sys
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
WORK = "/tmp/benchstrip"; os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
OUTDIR = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/benchmark_tube/data"
os.makedirs(OUTDIR, exist_ok=True)

W = 1.0
HR = [0.01, 0.03, 0.06, 0.12, 0.20]
NC, NR = 120, 16                         # NR even (NR/2 per ply for [45/-45])
ISO = dict(E=[70e9, 70e9, 70e9], G=[26.923e9]*3, nu=[0.3, 0.3, 0.3])
ANI = dict(E=[37e9, 9e9, 9e9], G=[4e9, 4e9, 4e9], nu=[0.3, 0.3, 0.3])


def gen_yaml(path, H, mat, layup):
    """layup: list of (angle_deg, frac_lo, frac_hi) through-thickness bands,
    bottom (frac 0) to top (frac 1)."""
    y2 = np.linspace(-W/2, W/2, NC + 1)
    y3 = np.linspace(-H/2, H/2, NR + 1)
    nid = lambda i, k: i*(NC + 1) + k          # i over thickness, k over width
    nodes, elems, oris = [], [], []
    for i in range(NR + 1):
        for k in range(NC + 1):
            nodes.append((y2[k], y3[i]))
    for i in range(NR):
        fr = (i + 0.5)/NR                       # band fraction (0 bottom, 1 top)
        thf = np.deg2rad(next(a for a, lo, hi in layup if lo <= fr < hi))
        cf, sf = np.cos(thf), np.sin(thf)
        for k in range(NC):
            a, b = nid(i, k), nid(i, k+1)
            c, d = nid(i+1, k+1), nid(i+1, k)
            elems.append((a+1, b+1, c+1, d+1))
            # EE1=fiber=(sf,0,cf), EE2=(cf,0,-sf), EE3=(0,1,0)  [+y3 through-thickness]
            oris.append([sf, 0.0, cf, cf, 0.0, -sf, 0.0, 1.0, 0.0])
    with open(path, "w") as f:
        f.write("nodes:\n")
        for y2v, y3v in nodes:
            f.write(f" - [{y2v:.10f} {y3v:.10f} 0.0]\n")
        f.write("elements:\n")
        for e in elems:
            f.write(f" - [{e[0]} {e[1]} {e[2]} {e[3]}]\n")
        f.write("sets:\n  element:\n  - name: strip\n    labels: ["
                + ", ".join(str(j+1) for j in range(len(elems))) + "]\n")
        f.write("materials:\n - name: strip\n")
        f.write(f"   E: {mat['E']}\n   G: {mat['G']}\n   nu: {mat['nu']}\n   rho: 1800.0\n")
        f.write("elementOrientations:\n")
        for o in oris:
            f.write(" - " + str([round(float(v), 10) for v in o]) + "\n")


def main():
    from opensg.mesh.segment import SolidBounMesh
    from opensg.core.solid import compute_timo_boun
    rows = []
    cases = [("iso", ISO, [(0.0, 0.0, 1.0)]),
             ("aniso", ANI, [(-45.0, 0.0, 0.5), (45.0, 0.5, 1.0)])]
    for matname, mat, layup in cases:
        for hr in HR:
            H = hr*W
            yml = os.path.join(WORK, f"strip_{matname}_{hr}.yaml")
            gen_yaml(yml, H, mat, layup)
            sm = SolidBounMesh(yml)
            mp, dens = sm.material_database
            C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
            rows.append([matname, hr] + list(C6.flatten()))
            print(f"[strip-solid] {matname} h/W={hr}: EA={C6[0,0]:.4e} GJ={C6[3,3]:.4e} "
                  f"EI2={C6[4,4]:.4e} EI3={C6[5,5]:.4e} GA2={C6[1,1]:.4e} GA3={C6[2,2]:.4e}")
    hdr = "material,hr," + ",".join(f"C{i+1}{j+1}" for i in range(6) for j in range(6))
    with open(os.path.join(OUTDIR, "strip_solid_6x6.csv"), "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("wrote", os.path.join(OUTDIR, "strip_solid_6x6.csv"))


if __name__ == "__main__":
    main()
