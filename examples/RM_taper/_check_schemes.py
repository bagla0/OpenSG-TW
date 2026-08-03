"""_check_schemes.py -- settle the shear scheme. TAPER = full integration (6-DOF) for ALL
cases. For the BOUNDARY ring, test whether the thin=gamma23-tie / thick=full rule works for
every geometry, by printing thin-boundary %err under mitc4_g23 vs full vs mitc4_both."""
import os
import numpy as np
import _rm_common as rm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "check")
C = rm.CC
B = os.path.join(C, "examples", "data", "benchmark")
LBL = rm.LBL


def diag_err(sh, so):
    sh = 0.5 * (np.asarray(sh) + np.asarray(sh).T); so = 0.5 * (np.asarray(so) + np.asarray(so).T)
    return 100 * (np.diag(sh) - np.diag(so)) / np.diag(so)


def fmt(e):
    return " ".join("%+6.1f" % v for v in e)


GEOM = {
    "circle": (os.path.join(C, "examples", "data", "taper_study", "meshes"),
               np.load(os.path.join(B, "taper_study_solid_m45.npz")), "%s_m45_aR070"),
    "square": (os.path.join(C, "examples", "data", "taper_square", "meshes"),
               np.load(os.path.join(B, "taper_square_solid_m45.npz")), "%s_m45_aR070"),
    "ellipse": (os.path.join(C, "examples", "data", "rm_taper_ellipse", "meshes"),
                np.load(os.path.join(B, "ellipse_solid_m45.npz")), "%s_m45"),
}

print("            terms:  %s" % " ".join("%6s" % l for l in LBL))
for g, (mesh_dir, ref, tgfmt) in GEOM.items():
    for regime, tR in [("thin", 0.02), ("thick", 0.20)]:
        tg = tgfmt % regime
        solL = ref[tg + "_L"]; solseg = ref[tg + "_seg"]
        print("\n=== %s %s (t/R=%.2f) ===" % (g, regime, tR))
        Ct = rm.solve_taper(mesh_dir, tg, RES, "full")
        print("  TAPER  full        : %s" % fmt(diag_err(Ct, solseg)))
        schemes = ["mitc4_g23", "full", "mitc4_both"] if regime == "thin" else ["full"]
        for sc in schemes:
            Cb = rm.solve_boundary(mesh_dir, tg, RES, sc)
            print("  BOUN   %-11s: %s" % (sc, fmt(diag_err(Cb, solL))))
print("\ndone", flush=True)
