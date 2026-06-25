"""
FEniCS 2D-solid (VABS) Timoshenko 6x6 for the isotropic strip over a THICKNESS
sweep at FIXED width W -- the reference for the thin-shell -> thick-shell study.
As h grows (h/W up) the wall stops being a thin shell.  Run in WSL (opensg_env_v8).
Saves data/strip_thickness_solid.csv.  Order [ext, shear2, shear3, twist, bend2, bend3].
"""
import os, sys
import numpy as np
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
WORK = "/tmp/stripthick"; os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
OUTDIR = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/benchmark_tube/data"
os.makedirs(OUTDIR, exist_ok=True)

W = 1.0                                     # fixed width
THICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]   # h/W = 0.01 (thin) .. 0.5 (thick)
NC, NR = 160, 16
ISO = dict(E=[70e9, 70e9, 70e9], G=[26.923e9]*3, nu=[0.3, 0.3, 0.3])


def gen_yaml(path, H):
    y2 = np.linspace(-W/2, W/2, NC+1)
    y3 = np.linspace(-H/2, H/2, NR+1)
    nid = lambda i, k: i*(NC+1) + k
    nodes = [(y2[k], y3[i]) for i in range(NR+1) for k in range(NC+1)]
    elems, oris = [], []
    for i in range(NR):
        for k in range(NC):
            a, b = nid(i, k), nid(i, k+1)
            c, d = nid(i+1, k+1), nid(i+1, k)
            elems.append((a+1, b+1, c+1, d+1))
            oris.append([0., 0., 1., 1., 0., 0., 0., 1., 0.])
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
        f.write(f"   E: {ISO['E']}\n   G: {ISO['G']}\n   nu: {ISO['nu']}\n   rho: 1800.0\n")
        f.write("elementOrientations:\n")
        for o in oris:
            f.write(" - " + str([round(float(v), 10) for v in o]) + "\n")


def main():
    from opensg.mesh.segment import SolidBounMesh
    from opensg.core.solid import compute_timo_boun
    rows = []
    for H in THICKS:
        yml = os.path.join(WORK, f"strip_h{H}.yaml")
        gen_yaml(yml, H)
        sm = SolidBounMesh(yml)
        mp, _ = sm.material_database
        C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
        rows.append([H] + list(C6.flatten()))
        print(f"[solid] h={H} (h/W={H/W:.3f}): EA={C6[0,0]:.4e} GA2={C6[1,1]:.4e} "
              f"GJ={C6[3,3]:.4e} EI2={C6[4,4]:.4e} EI3={C6[5,5]:.4e}")
    hdr = "h," + ",".join(f"C{i+1}{j+1}" for i in range(6) for j in range(6))
    with open(os.path.join(OUTDIR, "strip_thickness_solid.csv"), "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("wrote", os.path.join(OUTDIR, "strip_thickness_solid.csv"))


if __name__ == "__main__":
    main()
