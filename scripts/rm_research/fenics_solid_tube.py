"""
FEniCS-solid dehom of the anisotropic -45 deg tube (Opensg_MSG Table 3.2/3.3),
run in WSL (opensg_env_v8, dolfinx 0.8).  Generates a structured annulus quad
mesh + the 2D-solid YAML (fiber baked into EE1), homogenizes via compute_timo_boun
(should match Table 3.2), recovers the 3D material-frame stress via local_strain
under the SAME FF as the JAX run, and writes the y2=0, y3>0 (top wall) path.
"""
import os, sys
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
WORK = "/tmp/tube_solid"; os.makedirs(WORK, exist_ok=True); os.chdir(WORK)

R, ANG = 0.0715, -45.0
H = float(os.environ.get("TUBE_H", 0.008682))
NC = int(sys.argv[1]) if len(sys.argv) > 1 else 160   # circumferential divisions
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 12    # radial (through-thickness)
# optional drive: "force" (default) or a 6-vector beam strain "strain a b c d e f"
DRIVE = sys.argv[3] if len(sys.argv) > 3 else "force"
FF = np.array([2.0e4, 0.0, 0.0, 3.0e2, 1.5e2, 0.0])
if len(sys.argv) > 4:
    FF = np.array([float(v) for v in sys.argv[4:10]])
MAT_E = [37e9, 9e9, 9e9]; MAT_G = [4e9, 4e9, 4e9]; MAT_NU = [0.3, 0.3, 0.3]
TAB = {"C11 EA": 47.785e6, "C12 ext-tw": -0.93755e6, "C22 GJ": 0.14896e6, "C33 EI": 0.10710e6}


def gen_yaml(path):
    th_f = np.deg2rad(ANG)
    ri, ro = R - H/2, R + H/2
    rad = np.linspace(ri, ro, NR + 1)
    ang = np.array([2*np.pi*k/NC for k in range(NC)])
    nid = lambda i, k: i*NC + (k % NC)            # 0-based node id
    nodes, elems, oris = [], [], []
    for i in range(NR + 1):
        for k in range(NC):
            y2, y3 = rad[i]*np.cos(ang[k]), rad[i]*np.sin(ang[k])
            nodes.append((y2, y3))
    for i in range(NR):
        for k in range(NC):
            a, b = nid(i, k), nid(i, k+1)
            c, d = nid(i+1, k+1), nid(i+1, k)
            elems.append((a+1, b+1, c+1, d+1))     # 1-based, CCW quad
            t = ang[k] + np.pi/NC                   # element mid angle
            ct, st = np.cos(t), np.sin(t)
            cf, sf = np.cos(th_f), np.sin(th_f)
            # EE1 fiber = cos(th_f) beam + sin(th_f) tangent;  tangent=(-st,ct)
            o = [sf*(-st), sf*ct, cf,               # EE1 . (y2,y3,y1)
                 cf*(-st), cf*ct, -sf,              # EE2
                 ct, st, 0.0]                       # EE3 radial-out
            oris.append(o)
    with open(path, "w") as f:
        f.write("nodes:\n")
        for (y2, y3) in nodes:
            f.write(f" - [{y2:.10f} {y3:.10f} 0.0]\n")
        f.write("elements:\n")
        for e in elems:
            f.write(f" - [{e[0]} {e[1]} {e[2]} {e[3]}]\n")
        f.write("sets:\n  element:\n  - name: tube\n    labels: ["
                + ", ".join(str(j+1) for j in range(len(elems))) + "]\n")
        f.write("materials:\n - name: tube\n")
        f.write(f"   E: {MAT_E}\n   G: {MAT_G}\n   nu: {MAT_NU}\n   rho: 1800.0\n")
        f.write("elementOrientations:\n")
        for o in oris:
            f.write(" - " + str([round(float(v), 10) for v in o]) + "\n")
    return len(nodes), len(elems)


def main():
    yml = os.path.join(WORK, "aniso_tube_solid.yaml")
    nn, ne = gen_yaml(yml)
    print(f"mesh: {nn} nodes, {ne} quads (NC={NC}, NR={NR})")

    from opensg.mesh.segment import SolidBounMesh
    from opensg.core.solid import compute_timo_boun
    import opensg.core.stress_recov as stress_recov

    sm = SolidBounMesh(yml)
    mat_param, density = sm.material_database
    meshdata = sm.meshdata
    timo = compute_timo_boun(mat_param, meshdata)
    C6 = np.asarray(timo[0])
    np.savetxt(os.path.join(WORK, f"solid_C6_h{H:.6f}.txt"), C6)
    print("\nFEniCS-solid 6x6 (Deff_srt):\n", C6)
    print("\nFEniCS-solid homogenization vs Table 3.2:")
    print(f"  {'term':12s}{'solid':>13s}{'Table 3.2':>13s}{'% err':>9s}")
    for nm, (i, j) in [("C11 EA", (0, 0)), ("C12 ext-tw", (0, 3)),
                       ("C22 GJ", (3, 3)), ("C33 EI", (4, 4))]:
        v = TAB[nm]
        print(f"  {nm:12s}{C6[i,j]:13.4e}{v:13.4e}{100*(C6[i,j]-v)/v:9.1f}")

    beam_out = ([[[0, 0.0] for _ in range(6)] for _ in range(3)], None)
    drive_FF = FF
    if DRIVE == "strain":           # impose a pure beam strain: FF = C6 @ st
        drive_FF = C6 @ FF          # here FF holds the desired strain vector
        print("strain-driven: st =", FF.tolist())
    st_m, u_loc, strain_q, stress_q, coord_q = stress_recov.local_strain(
        timo, beam_out, 0, meshdata, mat_param, drive_FF)
    coords = coord_q.reshape(-1, 3)                 # [y1, y2, y3]
    stress = stress_q.x.array.reshape(-1, 6)        # [S11,S22,S33,S23,S13,S12]

    # path y2 ~ 0, y3 > 0 (top wall, through thickness)
    tol = np.pi*R/NC                                # ~ one element arc
    m = (np.abs(coords[:, 1]) < tol) & (coords[:, 2] > 0)
    yy = coords[m]; ss = stress[m]
    order = np.argsort(yy[:, 2])
    out = np.column_stack([yy[order][:, 1], yy[order][:, 2], ss[order]])
    tag = f"{DRIVE}_NC{NC}_NR{NR}_h{H:.6f}"
    if DRIVE == "strain":
        tag += "_" + "".join(f"{int(v)}" for v in FF)
    np.savetxt(os.path.join(WORK, f"tube_solid_toppath_{tag}.txt"), out,
               header="y2 y3 S11 S22 S33 S23 S13 S12", comments="")
    if DRIVE == "force":   # keep the canonical names for the default run
        np.savetxt(os.path.join(WORK, "tube_solid_toppath.txt"), out,
                   header="y2 y3 S11 S22 S33 S23 S13 S12", comments="")
        np.savetxt(os.path.join(WORK, "tube_solid_full.txt"),
                   np.column_stack([coords[:, 1], coords[:, 2], stress]),
                   header="y2 y3 S11 S22 S33 S23 S13 S12", comments="")
    s11 = out[:, 2]
    print(f"\n[CONV] NC={NC} NR={NR} drive={DRIVE}  EI={C6[4,4]:.4e}  "
          f"GJ={C6[3,3]:.4e}  S11_OML={s11[-1]/1e6:.3f}  S11_IML={s11[0]/1e6:.3f}  "
          f"grad={s11[-1]/1e6 - s11[0]/1e6:+.3f}  mean={s11.mean()/1e6:.3f} MPa  "
          f"(n={m.sum()})")
    sig = "".join(str(int(v)) for v in FF) if DRIVE == "strain" else "F"
    with open(os.path.join(WORK, "conv_metrics.csv"), "a") as fcsv:
        fcsv.write(f"{H:.6f},{DRIVE},{sig},{C6[4,4]:.6e},{C6[0,0]:.6e},"
                   f"{s11[-1]:.6e},{s11[0]:.6e},{s11.mean():.6e}\n")


if __name__ == "__main__":
    main()
