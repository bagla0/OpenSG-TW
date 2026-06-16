"""
Flat plate-strip (2D rectangle W x h) shell Timoshenko 6x6 (RM + Kirchhoff),
isotropic and anisotropic [45/-45], width W=1 m, over 5 thickness ratios h/W,
at both the OML (frac=0, top surface) and centre (frac=0.5, mid-surface)
references.  Saves data/strip_shell_6x6.csv.
Order [ext, shear2, shear3, twist, bend2, bend3] = [EA, GA12, GA13, GJ, EI2, EI3].

Geometry: shell reference line along y2 in [-W/2, W/2] at the top OML (y3=H/2);
the laminate stacks DOWN (e3 = (0,-1) inward), so frac=0.5 puts the reference at
the mid-surface y3=0 -- the same origin as the centred solid rectangle.  Flat:
k22 = 0.  The [45/-45] stack is +45 on top, -45 on the bottom (matches the solid).
"""
import os, sys
import numpy as np
import yaml as _yaml
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "rm"))
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

W, N = 1.0, 161
HR = [0.01, 0.03, 0.06, 0.12, 0.20]
OUTDIR = os.path.join(HERE, "data"); os.makedirs(OUTDIR, exist_ok=True)
ISO = {"E": [70e9, 70e9, 70e9], "G": [26.923e9]*3, "nu": [0.3, 0.3, 0.3]}
ANI = {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9], "nu": [0.3, 0.3, 0.3]}


def gen_yaml(path, H, mat, layup):
    """layup: list of (angle, thick), stacked from the top OML inward (downward)."""
    y2 = np.linspace(-W/2.0, W/2.0, N)
    nodes = [[float(y), float(H/2.0), 0.0] for y in y2]      # top OML
    elements = [[k+1, k+2] for k in range(N-1)]              # open strip
    ori = [[0., 0., 1., 1., 0., 0., 0., -1., 0.] for _ in range(N-1)]  # e3=(0,-1) inward
    data = {"nodes": nodes, "elements": elements,
            "sets": {"element": [{"name": "strip", "labels": list(range(1, N))}]},
            "sections": [{"elementSet": "strip",
                          "layup": [["mat", float(t), float(a)] for a, t in layup]}],
            "materials": [{"name": "mat", "density": 1800.0, "elastic": mat}],
            "elementOrientations": ori}
    with open(path, "w") as f:
        _yaml.safe_dump(data, f)


def shell_6x6(yaml_path, H, frac):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.zeros(len(elems))                               # flat -> no curvature
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
            H = hr*W
            layup = [(a, f*H) for a, f in layfrac]
            yml = os.path.join(OUTDIR, f"strip_{matname}_{hr}.yaml")
            gen_yaml(yml, H, mat, layup)
            for refname, frac in [("OML", 0.0), ("center", 0.5)]:
                RM, KF = shell_6x6(yml, H, frac)
                for model, C6 in [("RM", RM), ("KF", KF)]:
                    rows.append([matname, hr, refname, model] + list(C6.flatten()))
                print(f"[strip] {matname} h/W={hr} {refname}: RM EA={RM[0,0]:.4e} "
                      f"EI2={RM[4,4]:.4e} EI3={RM[5,5]:.4e}  KF EA={KF[0,0]:.4e} "
                      f"EI2={KF[4,4]:.4e}")
    hdr = "material,hr,reference,model," + ",".join(f"C{i+1}{j+1}" for i in range(6) for j in range(6))
    with open(os.path.join(OUTDIR, "strip_shell_6x6.csv"), "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("wrote", os.path.join(OUTDIR, "strip_shell_6x6.csv"))


if __name__ == "__main__":
    main()
