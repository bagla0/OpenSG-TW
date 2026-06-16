"""
Compare the FSDT/Whitney transverse-shear stiffness (current default) against the
coupling-aware "MSG / no-shear-correction-factor" 2x2 form (coupled=True) in the
RM homogenization, for the four benchmark cases.  Also: an isotropic 5/6 Gh sanity
check, and a G13 != G23 demo where the two routes genuinely differ.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "rm"))
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness
import bench_shell as TUBE
import bench_strip_shell as STRIP

R, W = 1.0, 1.0


def rm_6x6(yaml_path, H, frac, geom, coupled):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    if geom == "tube":
        r_ref = R if frac == 0.5 else R + H/2
        k22 = -1.0/r_ref*np.ones(len(elems))
    else:
        k22 = np.zeros(len(elems))

    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac*float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"],
                                           mat_db, coupled=coupled)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    return np.asarray(RM), list(G_by.values())[0]


# ---- 1) isotropic plate sanity: FSDT vs MSG vs 5/6 G h -----------------------
print("=== isotropic single-ply plate transverse shear (h=0.1) ===")
E, nu, h = 70e9, 0.3, 0.1; G = E/(2*(1+nu))
mat = {"iso": {"E": [E, E, E], "G": [G, G, G], "nu": [nu, nu, nu]}}
Gf = transverse_shear_stiffness([h], [0.0], ["iso"], mat, coupled=False)[0]
Gm = transverse_shear_stiffness([h], [0.0], ["iso"], mat, coupled=True)[0]
print(f"  5/6 G h        = {5/6*G*h:.6e}")
print(f"  FSDT  G[0,0]   = {Gf[0,0]:.6e}   off-diag={Gf[0,1]:.2e}")
print(f"  MSG   G[0,0]   = {Gm[0,0]:.6e}   off-diag={Gm[0,1]:.2e}")

# ---- 2) G13 != G23 off-axis demo: where FSDT and MSG diverge -----------------
print("\n=== [45/-45], G13=5e9 != G23=3e9: FSDT (diagonal) vs MSG (coupled 2x2) ===")
matd = {"d": {"E": [37e9, 9e9, 9e9], "G": [4e9, 5e9, 3e9], "nu": [0.3, 0.3, 0.3]}}
th = [45.0, -45.0]; tk = [0.05, 0.05]; nm = ["d", "d"]
Gf = transverse_shear_stiffness(tk, th, nm, matd, coupled=False)[0]
Gm = transverse_shear_stiffness(tk, th, nm, matd, coupled=True)[0]
print(f"  FSDT G = [[{Gf[0,0]:.3e}, {Gf[0,1]:.3e}], [{Gf[1,0]:.3e}, {Gf[1,1]:.3e}]]")
print(f"  MSG  G = [[{Gm[0,0]:.3e}, {Gm[0,1]:.3e}], [{Gm[1,0]:.3e}, {Gm[1,1]:.3e}]]")
print(f"  diag diff: {100*(Gm[0,0]-Gf[0,0])/Gf[0,0]:+.2f}% / {100*(Gm[1,1]-Gf[1,1])/Gf[1,1]:+.2f}%; "
      f"MSG off-diag/diag = {Gm[0,1]/np.sqrt(Gm[0,0]*Gm[1,1]):+.3f}")

# ---- 3) benchmark cases: RM 6x6 FSDT vs MSG ---------------------------------
ORD = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
cases = [("tube", "iso", TUBE), ("tube", "aniso", TUBE),
         ("strip", "iso", STRIP), ("strip", "aniso", STRIP)]
print("\n=== RM 6x6: FSDT vs MSG transverse shear (centre ref) ===")
for geom, mat, mod in cases:
    matprop, layfrac = (mod.ISO, [(0.0, 1.0)]) if mat == "iso" else \
                       (mod.ANI, [(45.0, 0.5), (-45.0, 0.5)])
    for hr in [0.06, 0.20]:
        H = hr*(R if geom == "tube" else W)
        layup = [(a, f*H) for a, f in layfrac]
        yml = os.path.join(HERE, "data", f"gcmp_{geom}_{mat}_{hr}.yaml")
        mod.gen_yaml(yml, H, matprop, layup)
        Rf, Gf = rm_6x6(yml, H, 0.5, geom, coupled=False)
        Rm, Gm = rm_6x6(yml, H, 0.5, geom, coupled=True)
        gblock = (f"Gblk FSDT diag=[{Gf[0,0]:.3e},{Gf[1,1]:.3e}] "
                  f"MSG diag=[{Gm[0,0]:.3e},{Gm[1,1]:.3e}] off={Gm[0,1]:.1e}")
        diffs = [100*(Rm[i, i]-Rf[i, i])/Rf[i, i] if abs(Rf[i, i]) > 1 else 0.0 for i in range(6)]
        dmax = max(abs(d) for d in diffs)
        print(f"\n{geom}/{mat} h={hr}:  {gblock}")
        print("   RM 6x6 %diff (MSG vs FSDT): " +
              "  ".join(f"{ORD[i]}={diffs[i]:+.3f}" for i in range(6)) +
              f"   | max |diff|={dmax:.3f}%")
