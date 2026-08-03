"""Step 3 - e1/e2/e3 material-orientation PNGs (shell + matching 2D-solid panel).

One PNG per representative case into figures/.  The orientation plot is the
geometry / e3 sanity check required for every OpenSG run; it catches wall-normal
(e3) and traversal bugs.

The 2D-solid panel draws one arrow per solid element, so it is skipped for very
fine solid meshes (> MAX_SOLID_ELEM) to keep the plot fast -- those cases fall
back to a shell-only panel.  Orientation is a QA aid and is never fatal.
"""
import os

import yaml

from common import MESH, REF, FIG
from opensg_jax.fe_jax.orient_plot import plot_orient

MAX_SOLID_ELEM = 7000

# (case name, shell mesh in meshes/, solid mesh in reference/)
ORIENT = [
    ("single_rh02",      "shell_rh02.yaml",            "solid_rh02.yaml"),
    ("single_rh10",      "shell_rh10.yaml",            "solid_rh10.yaml"),
    ("2cell_iso_thin",   "tube2cell_thin.yaml",        "solid_tube2cell_thin.yaml"),
    ("2cell_iso_thick",  "tube2cell_thick.yaml",       "solid_tube2cell_thick.yaml"),
    ("2cell_aniso_thin", "tube2cell_aniso_thin.yaml",  "solid_tube2cell_aniso_thin.yaml"),
    ("2cell_aniso_thick","tube2cell_aniso_thick.yaml", "solid_tube2cell_aniso_thick.yaml"),
]


def n_solid_elem(path):
    try:
        return len(yaml.safe_load(open(path)).get("elements", []))
    except Exception:
        return 10 ** 9                                   # unknown -> treat as "too big"


for name, shell, solid in ORIENT:
    sp = os.path.join(MESH, shell)
    solid_p = os.path.join(REF, solid)
    out = os.path.join(FIG, "orient_%s.png" % name)
    if not os.path.exists(sp):
        print("skip %s (missing %s)" % (name, sp))
        continue
    use_solid = os.path.exists(solid_p) and n_solid_elem(solid_p) <= MAX_SOLID_ELEM
    try:
        plot_orient(sp, solid_p if use_solid else None, out, side_by_side=True)
        tag = "shell+solid" if use_solid else "shell-only (solid mesh too fine to render quickly)"
        print("wrote %s   [%s]" % (out, tag))
    except Exception as e:                               # never fatal
        try:
            plot_orient(sp, None, out)
            print("wrote %s   [shell-only fallback: %s]" % (out, e))
        except Exception as e2:
            print("FAILED %s: %s" % (name, e2))
print("orientation PNGs -> %s" % FIG)
