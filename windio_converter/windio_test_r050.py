"""Build the IEA-22 cross-section at r=0.5 from windIO, emit the OpenSG YAML, show the orientation,
and verify it homogenizes through RM and KL (proves the converter output is a valid OpenSG SG)."""
import os, sys
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section, emit_opensg_yaml
from fe_jax.orient_plot import plot_orient
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
out = os.path.join(CC, "windio_converter", "out", "iea22_r050.yaml")
os.makedirs(os.path.dirname(out), exist_ok=True)

blade = WindIOBlade(src)
cs = build_cross_section(blade, 0.5, mesh_size=0.01)
info = emit_opensg_yaml(cs, out)
print("emitted:", info)

png = out.replace(".yaml", "_orient.png")
plot_orient(out, None, png)
print("orient:", png)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
RM = np.asarray(rm_timoshenko_6x6(out, 0.0, orient=False)); RM = 0.5 * (RM + RM.T)
KL = np.asarray(gradient_junction_kirchhoff(out, frac=0.0, orient=False)[0]); KL = 0.5 * (KL + KL.T)
print("\nIEA-22 r=0.5  Timoshenko 6x6 diagonal (RM vs KL):")
for i in range(6):
    print("  %-4s  RM %.4e   KL %.4e   (rel diff %+.2f%%)"
          % (LBL[i], RM[i, i], KL[i, i], 100 * (KL[i, i] - RM[i, i]) / RM[i, i]))
