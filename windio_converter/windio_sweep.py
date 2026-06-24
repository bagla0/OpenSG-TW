"""Convert ALL airfoil stations of the IEA-22 blade windIO -> OpenSG 1D YAML and homogenize each
(RM & KL) as a robustness check across the span."""
import os, sys
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section, emit_opensg_yaml
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
OUT = os.path.join(CC, "windio_converter", "out"); os.makedirs(OUT, exist_ok=True)
blade = WindIOBlade(src)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
stations = [round(0.1 * k, 2) for k in range(1, 10)] + [0.95]
print("station  chord  twist  nodes elem sets webs   EA        GA3       EI3     | max|RM-KL|%")
for r in stations:
    try:
        cs = build_cross_section(blade, r, mesh_size=0.01)
        out = os.path.join(OUT, "iea22_r%03d.yaml" % round(r * 100))
        info = emit_opensg_yaml(cs, out)
        RM = 0.5 * (np.asarray(rm_timoshenko_6x6(out, 0.0, orient=False)) + np.asarray(rm_timoshenko_6x6(out, 0.0, orient=False)).T)
        KL = np.asarray(gradient_junction_kirchhoff(out, frac=0.0, orient=False)[0]); KL = 0.5 * (KL + KL.T)
        md = max(abs(100 * (KL[i, i] - RM[i, i]) / RM[i, i]) for i in range(6))
        print("  %.2f   %5.2f  %5.1f  %4d %4d  %d   %d    %.3e %.3e %.3e |  %.2f"
              % (r, cs["chord"], cs["twist"], info["n_nodes"], info["n_elems"], info["n_sets"],
                 info["n_webs"], RM[0, 0], RM[2, 2], RM[5, 5], md))
    except Exception as e:
        print("  %.2f   FAILED: %s" % (r, repr(e)[:90]))
print("DONE")
