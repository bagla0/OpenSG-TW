"""Shared setup + case list for the OpenSG-TW Timoshenko-6x6 reproduction package.

Reproduces the single-cell and two-cell tube tables of the ASC-2026 paper
("OpenSG: A Native JAX-MSG Framework for Thin-Walled Composite Beams") by
re-running the JAX Kirchhoff-Love (gradient/Hermite-C1) and Reissner-Mindlin
(MITC) shell models and comparing the 6x6 Timoshenko stiffness against the
FEniCS-2D-solid (VABS) reference.

Timoshenko order everywhere: [EA, GA2, GA3, GJ, EI2, EI3]
  C11=EA  C22=GA2  C33=GA3  C44=GJ  C55=EI2  C66=EI3 ;
  dominant couplings  C14=EA-GJ (extension-twist),  C25=GA2-EI2,  C36=GA3-EI3.

Two curvature paths (this is intentional, and matches the paper):
  * single-cell smooth tube -> EXACT hoop curvature k22 = -1/R.  A plain circle
    is a known smooth surface, so the curvature is imposed analytically
    (lib.tube_lib.homog, k22_mode="exact").
  * two-cell webbed tube    -> GEOMETRIC per-element curvature (mesh_curvature),
    because the internal web is a FLAT wall (k22~0) while the outer wall is
    curved; the public drivers gradient_junction_kirchhoff / rm_timoshenko_6x6
    (curved=True) compute k22 element-by-element from the mesh geometry.

The package is self-contained: it needs only a working OpenSG-TW checkout
(the folder that contains opensg_jax/fe_jax) and its conda environment
(jax x64 + pypardiso + numpy/scipy/pyyaml + matplotlib).  No hardcoded user
paths -- the repo root is located automatically by walking up from this file.
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(BASE, "lib")
MESH = os.path.join(BASE, "meshes")
REF = os.path.join(BASE, "reference")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")


def _repo_root(start):
    """Walk up from `start` until we find the OpenSG-TW root (has opensg_jax/fe_jax)."""
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "opensg_jax", "fe_jax")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(
        "OpenSG-TW repo root (a folder containing opensg_jax/fe_jax) was not found "
        "above %s. Place this package inside your OpenSG-TW checkout." % start)


ROOT = _repo_root(BASE)
for _p in (ROOT, LIB, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _d in (MESH, RES, FIG):
    os.makedirs(_d, exist_ok=True)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
ORDER_NOTE = "Timoshenko 6x6 order: [EA, GA2, GA3, GJ, EI2, EI3]"

# term label -> (row, col) in the symmetric 6x6
TERM_IJ = {"C11": (0, 0), "C22": (1, 1), "C33": (2, 2), "C44": (3, 3),
           "C55": (4, 4), "C66": (5, 5), "C14": (0, 3), "C25": (1, 4), "C36": (2, 5)}

# ---------------- geometry / material ----------------
R_SINGLE = 0.0715                                  # single-cell mean (mid-wall) radius [m]
N_SINGLE = 3200                                    # circumferential segments (refined)
RH_LIST = list(range(1, 11))                       # R/h = 1 .. 10
ANI_MAT = {"E": [37.0e9, 9.0e9, 9.0e9],            # ud_frp orthotropic
           "G": [4.0e9, 4.0e9, 4.0e9],
           "nu": [0.28, 0.28, 0.28]}

R_TWOCELL = 0.05                                   # two-cell mean radius [m]
TWOCELL = [
    ("2cell_iso_thin",    "tube2cell_thin.yaml",        0.004, "C6_solid_tube2cell_thin.txt"),
    ("2cell_iso_thick",   "tube2cell_thick.yaml",       0.016, "C6_solid_tube2cell_thick.txt"),
    ("2cell_aniso_thin",  "tube2cell_aniso_thin.yaml",  0.004, "C6_solid_tube2cell_aniso_thin.txt"),
    ("2cell_aniso_thick", "tube2cell_aniso_thick.yaml", 0.016, "C6_solid_tube2cell_aniso_thick.txt"),
]


def cases():
    """Every case: single-cell R/h=1..10 (exact k22) + two-cell iso/aniso thin/thick (geometric k22)."""
    out = []
    for rh in RH_LIST:
        h = R_SINGLE / rh
        out.append(dict(name="single_rh%02d" % rh, kind="single", method="exact",
                        mesh=os.path.join(MESH, "shell_rh%02d.yaml" % rh),
                        R=R_SINGLE, dshift=h / 2.0, h=h, rh=rh,
                        solid=os.path.join(REF, "C6_solid_rh%02d.txt" % rh)))
    for name, meshfn, t, solidfn in TWOCELL:
        out.append(dict(name=name, kind="twocell", method="geometric",
                        mesh=os.path.join(MESH, meshfn),
                        R=R_TWOCELL, dshift=t / 2.0, t=t,
                        solid=os.path.join(REF, solidfn)))
    return out


def sym(M):
    M = np.asarray(M, float)
    return 0.5 * (M + M.T)


def pe(m, s):
    return 100.0 * (m - s) / s


def save_dat(path, M6, header):
    np.savetxt(path, sym(M6), fmt="%.8e", header=header + "\n" + ORDER_NOTE)


def load6(path):
    return sym(np.loadtxt(path))


def compute_kl(case):
    """JAX Kirchhoff-Love (gradient-junction / Hermite-C1) Timoshenko 6x6."""
    import jax
    jax.config.update("jax_enable_x64", True)
    if case["method"] == "exact":
        import tube_lib as T                       # smooth tube -> exact k22 = -1/R
        _RM, KF = T.homog(case["mesh"], case["R"], case["dshift"], k22_mode="exact")
        return sym(KF)
    from opensg_jax.fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff
    KF = gradient_junction_kirchhoff(case["mesh"], frac=0.0, dshift=case["dshift"], orient=False)[0]
    return sym(KF)


def compute_rm(case):
    """JAX Reissner-Mindlin (MITC, shear='mitc_both') Timoshenko 6x6."""
    import jax
    jax.config.update("jax_enable_x64", True)
    if case["method"] == "exact":
        import tube_lib as T                       # smooth tube -> exact k22 = -1/R
        RM, _KF = T.homog(case["mesh"], case["R"], case["dshift"], k22_mode="exact")
        return sym(RM)
    from opensg_jax.fe_jax.strip_RM import rm_timoshenko_6x6
    RM = rm_timoshenko_6x6(case["mesh"], 0.0, dshift=case["dshift"], curved=True, orient=False)
    return sym(RM)
