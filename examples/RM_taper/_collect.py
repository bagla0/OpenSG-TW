"""_collect.py -- run all 12 RM_taper cases (3 geom x 2 thickness x {boundary,taper}) with the
shear rule and dump shell + solid 6x6 to rm_taper_results.npz for building the paper tables."""
import os
import numpy as np
import _rm_common as rm

OUT = {}
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "collect")


def do(name, mesh_dir, cases):
    for regime, tR, tg, solL, solseg in cases:
        sb = rm.shear_for("boundary", tR); st = rm.shear_for("taper", tR)
        Cb = rm.solve_boundary(mesh_dir, tg, RES, sb)
        Ct = rm.solve_taper(mesh_dir, tg, RES, st)
        pre = "%s_%s" % (name, regime)
        OUT[pre + "_boun_shell"] = 0.5 * (Cb + Cb.T)
        OUT[pre + "_boun_solid"] = 0.5 * (np.asarray(solL) + np.asarray(solL).T)
        OUT[pre + "_taper_shell"] = 0.5 * (Ct + Ct.T)
        OUT[pre + "_taper_solid"] = 0.5 * (np.asarray(solseg) + np.asarray(solseg).T)
        print("done %s boun(%s) taper(%s)" % (pre, sb, st), flush=True)


C = rm.CC
circ = np.load(os.path.join(C, "examples", "data", "benchmark", "taper_study_solid_m45.npz"))
sq = np.load(os.path.join(C, "examples", "data", "benchmark", "taper_square_solid_m45.npz"))
el = np.load(os.path.join(C, "examples", "data", "benchmark", "ellipse_solid_m45.npz"))
do("circle", os.path.join(C, "examples", "data", "taper_study", "meshes"),
   [("thin", 0.02, "thin_m45_aR070", circ["thin_m45_aR070_L"], circ["thin_m45_aR070_seg"]),
    ("thick", 0.20, "thick_m45_aR070", circ["thick_m45_aR070_L"], circ["thick_m45_aR070_seg"])])
do("square", os.path.join(C, "examples", "data", "taper_square", "meshes"),
   [("thin", 0.02, "thin_m45_aR070", sq["thin_m45_aR070_L"], sq["thin_m45_aR070_seg"]),
    ("thick", 0.20, "thick_m45_aR070", sq["thick_m45_aR070_L"], sq["thick_m45_aR070_seg"])])
do("ellipse", os.path.join(C, "examples", "data", "rm_taper_ellipse", "meshes"),
   [("thin", 0.02, "thin_m45", el["thin_m45_L"], el["thin_m45_seg"]),
    ("thick", 0.20, "thick_m45", el["thick_m45_L"], el["thick_m45_seg"])])
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rm_taper_results.npz"), **OUT)
print("SAVED", len(OUT), "arrays", flush=True)
