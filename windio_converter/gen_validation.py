"""Generate IEA-22 validation meshes at several stations: 1D-shell SG YAML (for RM/KL) + PreVABS 2D-solid
(.sg -> 2D-solid YAML for FEniCS). Windows: build+emit (py) -> prevabs.exe -> convert_sg_to_yaml."""
import os, sys, subprocess
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "windio_converter"))
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section, emit_opensg_yaml, emit_prevabs

PREVABS = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-windows-x64-full\prevabs-v2.0.1-windows-x64-full\prevabs.exe"
CONVERT = os.path.join(CC, "prevabs_mh104", "convert_sg_to_yaml.py")
PY = r"C:\conda_envs\opensg_2_0_env\python.exe"
src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")
VAL = os.path.join(CC, "windio_converter", "validation"); os.makedirs(VAL, exist_ok=True)

STATIONS = [float(x) for x in (sys.argv[1:] or [0.3, 0.5, 0.7])]
SHELL_MS, SOLID_MS = 0.01, 0.02
blade = WindIOBlade(src)
for r in STATIONS:
    tag = "r%03d" % round(r * 100)
    cs = build_cross_section(blade, r, mesh_size=SHELL_MS)
    emit_opensg_yaml(cs, os.path.join(VAL, "shell_iea22_%s.yaml" % tag), )
    pv = os.path.join(VAL, "prevabs_%s" % tag)
    emit_prevabs(cs, pv, name="iea22_%s" % tag, mesh_size=SOLID_MS)
    subprocess.run([PREVABS, "-i", "iea22_%s.xml" % tag, "--vabs", "--hm"], cwd=pv, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sg = os.path.join(pv, "iea22_%s.sg" % tag)
    solid_yaml = os.path.join(VAL, "solid_iea22_%s.yaml" % tag)
    subprocess.run([PY, CONVERT, sg, solid_yaml], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import yaml
    d = yaml.safe_load(open(solid_yaml))
    print("r=%.2f chord=%.3f  shell:%d elems  solid:%d nodes/%d elems  [%d laminates,%d webs]"
          % (r, cs["chord"], len(cs["elems"]), len(d["nodes"]), len(d["elements"]),
             len(cs["laminates"]), len(cs["webs"])), flush=True)
print("DONE -> validation/")
