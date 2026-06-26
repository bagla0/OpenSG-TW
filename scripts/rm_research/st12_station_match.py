"""
The 2Dsolid_12.yaml solid sits at beam_x=44.8276 = 13/29*100 = station 13.
Find which 1Dshell station (10,11,12,13) actually matches that solid: EA is
reference/origin-independent, so it pins the station; then report ALL non-zero
6x6 terms for the matching station. Shells evaluated at the centroid reference.
Order [ext, shear2, shear3, twist, bend2, bend3].
"""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import (read_mesh, mesh_curvature, offset_oml_to_iml,
                             element_e3_from_yaml)
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

SHELLDIR = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG"
SOLID = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
         r"\outputs\st12\solid_C6_st12.txt")
LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]
NZ = [(0, 0), (0, 4), (0, 5), (1, 1), (1, 2), (1, 3),
      (2, 2), (2, 3), (3, 3), (4, 4), (4, 5), (5, 5)]


def shell_6x6(yaml_path, frac=0.5):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    n3d = n3d.copy(); n3d[:, 2] = 0.0
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    y2b = (nodes[:, 0].min(), nodes[:, 0].max()); y3b = (nodes[:, 1].min(), nodes[:, 1].max())
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac*float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, KF, _ = timoshenko_from_yaml(yaml_path, frac=frac)
    return np.asarray(RM), np.asarray(KF), y2b, y3b


def main():
    S = np.loadtxt(SOLID)
    print(f"solid (2Dsolid_12.yaml @ beam_x=44.828 = station 13)  EA={S[0,0]:.4e}\n")
    print(f"  {'shell':8s}{'EA':>13s}{'EA %vs solid':>14s}{'6x6 norm %':>12s}"
          f"{'y2 range':>20s}")
    results = {}
    for N in [10, 11, 12, 13]:
        RM, KF, y2b, y3b = shell_6x6(os.path.join(SHELLDIR, f"1Dshell_{N}.yaml"))
        ea = RM[0, 0]
        sr = sum((RM[i, j]-S[i, j])**2 for i, j in NZ)
        sv = sum(S[i, j]**2 for i, j in NZ)
        norm = 100*(sr/sv)**0.5
        results[N] = (RM, KF, norm)
        print(f"  1Dshell_{N:<2d}{ea:13.4e}{100*(ea-S[0,0])/S[0,0]:13.1f}%"
              f"{norm:11.1f}%   [{y2b[0]:.3f},{y2b[1]:.3f}]")

    best = min(results, key=lambda k: results[k][2])
    RM, KF, _ = results[best]
    print(f"\n==> best match: station {best}  (the solid's true station)\n")
    print(f"All non-zero 6x6 terms, solid vs 1Dshell_{best} (centroid):")
    print(f"  {'i,j':7s}{'term':12s}{'solid':>14s}{'RM':>14s}{'KF':>14s}{'RM%':>8s}{'KF%':>8s}")
    sr = sk = sv = 0.0
    for i, j in NZ:
        v = S[i, j]; nm = f"{LBL[i][:3]}-{LBL[j][:3]}"
        rmp = 100*(RM[i, j]-v)/v if abs(v) > 1 else float('nan')
        kfp = 100*(KF[i, j]-v)/v if abs(v) > 1 else float('nan')
        print(f"  {i+1},{j+1:<4d}{nm:12s}{v:14.4e}{RM[i,j]:14.4e}{KF[i,j]:14.4e}"
              f"{rmp:8.1f}{kfp:8.1f}")
        sr += (RM[i, j]-v)**2; sk += (KF[i, j]-v)**2; sv += v**2
    print(f"\n  relative norm:  RM {100*(sr/sv)**0.5:.2f}%   Kirchhoff {100*(sk/sv)**0.5:.2f}%")


if __name__ == "__main__":
    main()
