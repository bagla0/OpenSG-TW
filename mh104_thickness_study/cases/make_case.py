"""Create cases/f0NN for an arbitrary thickness factor (scale f020 lamina by factor/0.2) and run
PreVABS to get mh104.sg.  Usage: python make_case.py 0.30 0.75 ..."""
import os
import re
import shutil
import subprocess
import sys

STUDY = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mh104_thickness_study"
SRC = os.path.join(STUDY, "cases", "f020")
PV = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-windows-x64-full\prevabs-v2.0.1-windows-x64-full\prevabs.exe"

for arg in sys.argv[1:]:
    f = float(arg); scale = f / 0.2
    dst = os.path.join(STUDY, "cases", "f%03d" % int(round(f * 100)))
    os.makedirs(dst, exist_ok=True)
    for fn in ("mh104.xml", "mh104.dat"):
        shutil.copy(os.path.join(SRC, fn), os.path.join(dst, fn))
    mat = open(os.path.join(SRC, "materials.xml")).read()
    mat = re.sub(r"<thickness>([0-9.eE+\-]+)</thickness>",
                 lambda m: "<thickness>%g</thickness>" % (float(m.group(1)) * scale), mat)
    open(os.path.join(dst, "materials.xml"), "w").write(mat)
    r = subprocess.run([PV, "--input", "mh104.xml", "--vabs", "--hm"], cwd=dst, capture_output=True, text=True)
    ok = os.path.exists(os.path.join(dst, "mh104.sg"))
    print("f=%.2f -> %s   prevabs rc=%d  mh104.sg=%s" % (f, dst, r.returncode, ok), flush=True)
