"""Create the f=0.10 case (f020 lamina thicknesses x0.5 = base x0.1) and run PreVABS to get mh104.sg."""
import os
import re
import shutil
import subprocess

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
src = os.path.join(STUDY, "cases", "f020")
dst = os.path.join(STUDY, "cases", "f010")
PV = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-windows-x64-full\prevabs-v2.0.1-windows-x64-full\prevabs.exe"

os.makedirs(dst, exist_ok=True)
for fn in ("mh104.xml", "mh104.dat"):
    shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
mat = open(os.path.join(src, "materials.xml")).read()
mat = re.sub(r"<thickness>([0-9.eE+\-]+)</thickness>",
             lambda m: "<thickness>%g</thickness>" % (float(m.group(1)) * 0.5), mat)
open(os.path.join(dst, "materials.xml"), "w").write(mat)
print("created cases/f010 (lamina x0.5 of f020)")

r = subprocess.run([PV, "mh104.xml", "-h"], cwd=dst, capture_output=True, text=True)
print("prevabs rc =", r.returncode)
tail = (r.stdout or "")[-700:]
print(tail)
if r.stderr:
    print("STDERR:", r.stderr[-400:])
print("mh104.sg exists:", os.path.exists(os.path.join(dst, "mh104.sg")))
