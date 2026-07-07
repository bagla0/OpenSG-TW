"""verify_sparse_driver.py -- sparse shell segment driver vs the dense driver.
Must match to ~1e-8 (same formulation; only assembly + Dirichlet solve differ)."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import run_indep as ri

RES = os.path.join(HERE, "out", "vsp_res"); os.makedirs(RES, exist_ok=True)
for sub, tg in (("taper_square", "thin_iso_aR070"),
                ("taper_study", "thin_m45_aR070"),
                ("taper_square", "thick_m45_aR070")):
    md = os.path.join(HERE, "out", sub, "meshes")
    Sd = np.asarray(ri.shell_solve_lagrange(tg, md, RES))
    Ss = np.asarray(ri.shell_solve_lagrange_sparse(tg, md, RES))
    rel = np.abs(Sd - Ss).max() / np.abs(Sd).max()
    print("%-14s %-16s relmax(sparse-dense) = %.3e   %s"
          % (sub, tg, rel, "OK" if rel < 1e-8 else "*** MISMATCH"))
