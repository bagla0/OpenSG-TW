"""Regenerate the IEA-22 shell YAMLs (after the web-direction fix) and re-plot solid+shell orientation."""
import os, sys
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section, emit_opensg_yaml
from fe_jax.orient_plot import plot_orient

src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
VAL = os.path.join(CC, "windio_converter", "validation")
blade = WindIOBlade(src)
for r, tag in ((0.3, "r030"), (0.5, "r050"), (0.7, "r070")):
    cs = build_cross_section(blade, r, mesh_size=0.01)
    sh = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    emit_opensg_yaml(cs, sh)
    plot_orient(sh, os.path.join(VAL, "solid_iea22_%s.yaml" % tag), os.path.join(VAL, "orient_iea22_%s.png" % tag))
print("regenerated shells + orientations")
