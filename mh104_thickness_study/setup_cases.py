"""mh104 wall-thickness sweep: scale ALL lamina thicknesses by factor f in {0.2..1.0}, re-run PreVABS
to get the scaled VABS .sg, convert to a 2D-solid YAML, and build the matching 1D-shell YAML (layups
scaled by the same f). This characterises the thin-wall convergence: as the walls thin (f->0.2) the
thin-wall shell should approach the 2D solid / VABS; at f=1.0 they diverge (the +40% over-stiffness).

The airfoil OML contour (mh104.dat) is fixed for all f -- only the wall thickness (inward of the OML)
scales -- so the solid OML and the shell OML contour stay identical across the sweep."""
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-examples\prevabs-v2.0.1-examples\ex_airfoil_r"
PV = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-windows-x64-full\prevabs-v2.0.1-windows-x64-full\prevabs.exe"
MH104 = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\prevabs_mh104"
PY = r"C:\conda_envs\opensg_2_0_env\python.exe"
FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0]

for sub in ("cases", "yaml_solid", "yaml_shell", "results", "plots"):
    os.makedirs(os.path.join(HERE, sub), exist_ok=True)


def scale_materials(src, dst, f):
    txt = open(src).read()
    txt = re.sub(r"<thickness>\s*([0-9.eE+-]+)\s*</thickness>",
                 lambda m: "<thickness>%.10g</thickness>" % (float(m.group(1)) * f), txt)
    open(dst, "w").write(txt)


def tag_of(f):
    return "f%03d" % int(round(f * 100))


for f in FACTORS:
    tag = tag_of(f)
    cdir = os.path.join(HERE, "cases", tag)
    os.makedirs(cdir, exist_ok=True)
    shutil.copy(os.path.join(SRC, "mh104.xml"), cdir)
    shutil.copy(os.path.join(SRC, "mh104.dat"), cdir)
    scale_materials(os.path.join(SRC, "materials.xml"), os.path.join(cdir, "materials.xml"), f)
    try:
        r = subprocess.run([PV, "-i", "mh104.xml", "--hm"], cwd=cdir, capture_output=True, text=True, timeout=300)
        sg = os.path.join(cdir, "mh104.sg")
        ok = os.path.exists(sg)
        nlines = sum(1 for _ in open(sg)) if ok else 0
        print("[%s] PreVABS %s  sg_lines=%d  %s" % (tag, "OK" if ok else "FAIL", nlines, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""))
        if not ok:
            print("   stderr:", r.stderr[-300:]); continue
        out2d = os.path.join(HERE, "yaml_solid", "solid_%s.yaml" % tag)
        c = subprocess.run([PY, os.path.join(MH104, "convert_sg_to_yaml.py"), sg, out2d], capture_output=True, text=True)
        print("   2D-solid:", "OK" if os.path.exists(out2d) else "FAIL", c.stdout.strip().splitlines()[-1] if c.stdout.strip() else c.stderr[-200:])
        out1d = os.path.join(HERE, "yaml_shell", "shell_%s.yaml" % tag)
        b = subprocess.run([PY, os.path.join(MH104, "build_mh104_shell_yaml.py"), str(f), out1d], capture_output=True, text=True)
        print("   1D-shell:", "OK" if os.path.exists(out1d) else "FAIL", b.stdout.strip().splitlines()[-1] if b.stdout.strip() else b.stderr[-200:])
    except Exception as e:
        print("[%s] ERROR %r" % (tag, e))
print("\nsetup complete.")
