"""rm_core.py -- the MINIMAL MSG-RM cross-section example: read a 1-D shell YAML,
homogenize it to a Timoshenko 6x6, and dehomogenize a beam load back to pointwise
3-D stress and displacement.

No plotting, no VABS comparison, no outlier masking -- just the three calls that
matter.  Everything is read from data committed in this repository, and every path
is resolved relative to this file, so it runs from a fresh clone at any location.

    python examples/RM_cross_section/rm_core.py

The three steps are:

  1. HOMOGENIZE   ring_6dof(load_ring(yaml))            -> Timoshenko 6x6  [EA,GA2,GA3,GJ,EI2,EI3]
  2. WALL LAW     rm_plate_msg(ply thicknesses/angles)  -> the MSG-RM 8x8 = blkdiag(ABD, G)
  3. DEHOMOGENIZE build_rm_bundle(yaml) + stress_at_points / disp_at_points

See docs/theory/reissner_mindlin.md and docs/theory/dehomogenization.md for the theory,
and docs/tutorials/iea_r020_homo_dehom.ipynb for the full validated study.
"""
import os
import sys
import time

import numpy as np
import yaml


# ---------------------------------------------------------------- repo + imports
def _repo_root(d=None):
    """Walk up from this file to the repository root (the dir holding examples/data)."""
    d = os.path.abspath(d or os.path.dirname(os.path.abspath(__file__)))
    while True:
        if os.path.isdir(os.path.join(d, "examples", "data")) and \
           os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("run this from inside the OpenSG-TW repository")
        d = p


CC = _repo_root()
for q in (CC, os.path.join(CC, "examples", "TW-paper", "xsec_paper"),
          os.path.join(CC, "mitc_rm_segment")):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # keep JAX on the CPU

import jax                                                  # noqa: E402
jax.config.update("jax_enable_x64", True)                   # MANDATORY: the KKT solve is float64

from xsec_5v6_master import load_ring, ring_6dof, LBL       # noqa: E402  homogenization
from msg_rm_plate import rm_plate_msg                       # noqa: E402  the MSG-RM wall law
from emit_abd import material_db_from_yaml                  # noqa: E402
import dehom_rm                                             # noqa: E402  dehomogenization

# ---------------------------------------------------------------- inputs (in-repo)
DATA = os.path.join(CC, "examples", "data", "iea_all_stations")
SHELL = os.path.join(DATA, "shell51", "1d_yaml", "iea_s10_shell.yaml")   # IEA-22, r/R = 0.2
FFDAT = os.path.join(DATA, "dehom51", "beamdyn", "ff51_rmc_reform.dat")  # per-station beam loads
STATION = 10                                                            # row of FFDAT -> r/R = 0.2


# ================================================================ 1. HOMOGENIZATION
def homogenize(shell_yaml):
    """1-D shell SG YAML -> Timoshenko 6x6 (VABS order [EA, GA2, GA3, GJ, EI2, EI3])."""
    ring = load_ring(shell_yaml)          # nodes, elements, per-layup ABD + wall shear G
    return np.asarray(ring_6dof(ring))    # the 6-DOF drilling-Lagrange ring solve


# ================================================================ 2. THE WALL LAW
def wall_8x8(shell_yaml, which=0, frac=0.5):
    """MSG-RM 8x8 = blkdiag(ABD 6x6, G 2x2) for ONE layup of the section.
    frac = 0.5 center / 0.0 OML / 1.0 IML reference."""
    d = yaml.safe_load(open(shell_yaml))
    mdb = material_db_from_yaml(d["materials"])      # NOTE: the materials LIST
    sec = d["sections"][which]
    lay = sec["layup"]                               # each ply = [material, thickness, angle]
    thk = [float(p[1]) for p in lay]
    ang = [float(p[2]) for p in lay]
    mat = [str(p[0]) for p in lay]
    r = rm_plate_msg(thk, ang, mat, mdb, n_per_layer=4, elem_order=3,
                     z_ref=frac * sum(thk))
    P = np.zeros((8, 8))
    P[:6, :6] = r["A6"]                              # A, B, D  (zeroth-order plate SG)
    P[6:, 6:] = r["G_msg"]                           # the VAM transverse-shear block
    return P, sec["elementSet"], r


# ================================================================ 3. DEHOMOGENIZATION
def dehomogenize(shell_yaml, points, beam_force):
    """Beam force/moment (VABS order) -> pointwise 3-D stress + warping displacement."""
    B = dehom_rm.build_rm_bundle(shell_yaml)         # ref read from the yaml; MSG wall G
    res = dehom_rm.stress_at_points(B, points, beam_force_vabs=beam_force, frame="material")
    disp = dehom_rm.disp_at_points(B, points, beam_force_vabs=beam_force)
    return B, res, disp


if __name__ == "__main__":
    print("station :", os.path.relpath(SHELL, CC))
    print("reference:", yaml.safe_load(open(SHELL)).get("reference"))

    # ---- 1. Timoshenko 6x6 -------------------------------------------------
    t0 = time.perf_counter()
    C6 = homogenize(SHELL)
    print("\n1. HOMOGENIZATION  (%.2f s)" % (time.perf_counter() - t0))
    for i in range(6):
        print("   %-4s = %12.5e" % (LBL[i], C6[i, i]))

    # ---- 2. the MSG-RM 8x8 wall law ---------------------------------------
    P, name, r = wall_8x8(SHELL, which=0)
    print("\n2. WALL LAW  (layup '%s')" % name)
    print("   A diag = %s  N/m" % np.array2string(np.diag(P)[:3], precision=3))
    print("   D diag = %s  N.m" % np.array2string(np.diag(P)[3:6], precision=3))
    print("   G      = %s  N/m   (LS residual %.1e)"
          % (np.array2string(np.diag(P)[6:], precision=4), r["Ustar_rel"]))

    # ---- 3. recovery at a few section points ------------------------------
    FF = np.loadtxt(FFDAT)[STATION, 1:]               # [F1,F2,F3,M1,M2,M3]
    ring = load_ring(SHELL)
    pts = np.asarray(ring["rx"])[:, :2][::80]         # a few contour points (y2, y3)
    t0 = time.perf_counter()
    B, res, disp = dehomogenize(SHELL, pts, FF)
    print("\n3. DEHOMOGENIZATION  (%.1f s, %d points)"
          % (time.perf_counter() - t0, len(pts)))
    print("   beam force FF =", np.array2string(FF, precision=3))
    print("   %9s %9s %11s %11s %11s %11s"
          % ("y2 [m]", "y3 [m]", "s11 [MPa]", "s22 [MPa]", "s12 [MPa]", "|u| [mm]"))
    S = np.asarray(res["stress"]); U = np.asarray(disp)
    for k in range(len(pts)):
        print("   %9.4f %9.4f %11.2f %11.3f %11.3f %11.3f"
              % (pts[k, 0], pts[k, 1], S[k, 0] / 1e6, S[k, 1] / 1e6, S[k, 5] / 1e6,
                 np.linalg.norm(U[k]) * 1e3))
    print("\nstress columns are Voigt [S11,S22,S33,S23,S13,S12] in the MATERIAL frame;")
    print("disp_at_points returns the WARPING only -- add the beam kinematics")
    print("u = u_g + C(w + r) - r for a total displacement (see docs/theory/dehomogenization.md).")
