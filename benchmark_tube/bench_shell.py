"""
Thin-walled shell Timoshenko 6x6 (RM + Kirchhoff) for the benchmark tubes,
isotropic and anisotropic [45/-45], R=1 m, over 5 h/R values, at both the OML
(frac=0) and centre (frac=0.5) references. Saves data/shell_6x6.csv.
Order [ext, shear2, shear3, twist, bend2, bend3].
"""
import os, sys
import numpy as np
import yaml as _yaml
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "rm"))
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import (read_mesh, offset_oml_to_iml, element_e3_from_yaml)
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

R, N = 1.0, 160
HR = [0.01, 0.03, 0.06, 0.12, 0.20]
OUTDIR = os.path.join(HERE, "data"); os.makedirs(OUTDIR, exist_ok=True)
ISO = {"E": [70e9, 70e9, 70e9], "G": [26.923e9]*3, "nu": [0.3, 0.3, 0.3]}
ANI = {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9], "nu": [0.3, 0.3, 0.3]}


def gen_yaml(path, H, mat, layup):
    """layup: list of (angle, thick). Nodes at OML (R+H/2). e3 inward."""
    Rg = R + H/2.0
    th = np.array([2*np.pi*k/N for k in range(N)])
    nodes = [[float(Rg*np.cos(t)), float(Rg*np.sin(t)), 0.0] for t in th]
    elements = [[k+1, k+2] for k in range(N-1)] + [[N, 1]]
    thm = np.array([np.pi*(2*k+1)/N for k in range(N)])
    ori = [[0., 0., 1., float(-np.sin(t)), float(np.cos(t)), 0.,
            float(-np.cos(t)), float(-np.sin(t)), 0.] for t in thm]
    data = {"nodes": nodes, "elements": elements,
            "sets": {"element": [{"name": "tube", "labels": list(range(1, N+1))}]},
            "sections": [{"elementSet": "tube",
                          "layup": [["mat", float(t), float(a)] for a, t in layup]}],
            "materials": [{"name": "mat", "density": 1800.0, "elastic": mat}],
            "elementOrientations": ori}
    with open(path, "w") as f:
        _yaml.safe_dump(data, f)


def shell_6x6(yaml_path, layup_db_thick, H, frac):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    r_ref = R if frac == 0.5 else (R + H/2)
    k22 = -1.0/r_ref * np.ones(len(elems))

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac*float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, KF, _ = timoshenko_from_yaml(yaml_path, frac=frac)
    return np.asarray(RM), np.asarray(KF)


def main():
    cases = [("iso", ISO, [(0.0, 1.0)]), ("aniso", ANI, [(45.0, 0.5), (-45.0, 0.5)])]
    rows = []
    for matname, mat, layfrac in cases:
        for hr in HR:
            H = hr*R
            layup = [(a, f*H) for a, f in layfrac]
            yml = os.path.join(OUTDIR, f"shell_{matname}_{hr}.yaml")
            gen_yaml(yml, H, mat, layup)
            for refname, frac in [("OML", 0.0), ("center", 0.5)]:
                RM, KF = shell_6x6(yml, None, H, frac)
                for model, C6 in [("RM", RM), ("KF", KF)]:
                    rows.append([matname, hr, refname, model] + list(C6.flatten()))
                print(f"[shell] {matname} h/R={hr} {refname}: "
                      f"RM EA={RM[0,0]:.4e} GJ={RM[3,3]:.4e} EI={RM[4,4]:.4e}  "
                      f"KF EA={KF[0,0]:.4e} GJ={KF[3,3]:.4e}")
    hdr = "material,hr,reference,model," + ",".join(f"C{i+1}{j+1}" for i in range(6) for j in range(6))
    with open(os.path.join(OUTDIR, "shell_6x6.csv"), "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("wrote", os.path.join(OUTDIR, "shell_6x6.csv"))


if __name__ == "__main__":
    main()
