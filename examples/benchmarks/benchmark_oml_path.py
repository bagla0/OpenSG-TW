"""
Exact-coordinate benchmark setup: evaluate MSG-TW stress on a chosen path and
write the path coordinates so the FEniCS solid can be sampled at the SAME points.

The path here = the upper-surface OML reference nodes (points that lie along the
1D TW mesh), nudged a small distance inside the wall so the solid point-evaluation
lands robustly inside the 2D mesh.  Because ``stress_at_points`` works at ANY
cross-section coordinate, the path can be replaced by any user coordinates.

Writes
  outputs/oml_path_coords.txt : (y2, y3) path coordinates  -> read by the WSL
                                FEniCS driver, which writes oml_fenics_atpath.txt
  outputs/oml_tw_atpath.txt   : TW global-frame stress at those coordinates

Then run the WSL FEniCS driver and benchmark_oml_excel.py (exact-coord mode).
"""
import os
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
sys.path.insert(0, os.path.dirname(__file__))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points
from benchmark_oml_compare import upper_te_to_le
from benchmark_oml_jax import YAML, FF

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
NUDGE = 0.005   # inward shift (toward section interior) so solid eval lands inside

bundle = solve_tw_from_yaml(YAML)
corners = np.asarray(bundle["corners"])
cen = corners.mean(axis=0)

# path = upper-surface reference nodes, TE -> LE, nudged inside the wall
idx = upper_te_to_le(corners)
path = corners[idx]
d = cen - path; d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-30)
path = path + NUDGE * d

# TW stress at those exact coordinates (global frame)
out = stress_at_points(bundle, path, beam_force_vabs=FF, frame="global")
print(f"path: {len(path)} pts, TW projected depth "
      f"[{out['depth'].min():.4f}, {out['depth'].max():.4f}] (target ~{NUDGE})")

os.makedirs(OUT, exist_ok=True)
hdr = ("FF=" + ",".join(f"{v:g}" for v in FF) + "\n"
       "y2 y3 S11 S22 S33 S23 S13 S12 (global frame, Pa)")
np.savetxt(os.path.join(OUT, "oml_path_coords.txt"), path, fmt="%18.8e")
np.savetxt(os.path.join(OUT, "oml_tw_atpath.txt"),
           np.column_stack([path, out["stress"]]), header=hdr, fmt="%18.8e")
print(f"wrote {os.path.join(OUT, 'oml_path_coords.txt')} and oml_tw_atpath.txt")
print("next: run the WSL FEniCS driver, then benchmark_oml_excel.py --exact")
