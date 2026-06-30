"""R/h convergence sweep (thesis Fig 3.6): build the 2D-solid annulus mesh for
R/h = 1..10 at fixed mean radius R=0.0715, single ply [-45].  PreVABS + convert.
Wall thickness h = R/(R/h);  OML radius = R + h/2;  mesh_size = h/8 (8 thru-thick)."""
import os
import subprocess

R = 0.0715
RH = list(range(1, 11))
PREVABS = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-windows-x64-full\prevabs-v2.0.1-windows-x64-full\prevabs.exe"
CONV = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\prevabs_mh104\convert_sg_to_yaml.py"
PY = r"C:\conda_envs\opensg_2_0_env\python.exe"
HERE = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\tube_thesis_314\sweep"
CASES = os.path.join(HERE, "cases")
DATA = os.path.join(HERE, "data")
os.makedirs(CASES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

XML = """<cross_section name="{name}">
  <analysis><model>1</model></analysis>
  <general><mesh_size>{ms:.7f}</mesh_size><element_type>linear</element_type></general>
  <component><segment><baseline>blcircle</baseline><layup direction="left">layup1</layup></segment></component>
  <baselines>
    <point name="ct">0 0</point>
    <line name="blcircle" type="circle"><center>ct</center><radius>{oml:.7f}</radius><discrete by="angle">1.8</discrete><direction>ccw</direction></line>
  </baselines>
  <materials>
    <material name="ud_frp" type="orthotropic"><density>1.86E+03</density><elastic>
      <e1>3.70E+10</e1><e2>9.00E+09</e2><e3>9.00E+09</e3>
      <g12>4.00E+09</g12><g13>4.00E+09</g13><g23>4.00E+09</g23>
      <nu12>0.28</nu12><nu13>0.28</nu13><nu23>0.28</nu23></elastic></material>
    <lamina name="la_full"><material>ud_frp</material><thickness>{h:.7f}</thickness></lamina>
  </materials>
  <layups><layup name="layup1" method="stack sequence"><lamina>la_full</lamina><code>[-45]</code></layup></layups>
</cross_section>
"""

for rh in RH:
    h = R / rh
    oml = R + h / 2.0
    ms = h / 8.0
    name = "tube_rh%02d" % rh
    open(os.path.join(CASES, name + ".xml"), "w").write(
        XML.format(name=name, ms=ms, oml=oml, h=h))
    r1 = subprocess.run([PREVABS, "--input", name + ".xml", "--vabs", "--hm"],
                        cwd=CASES, capture_output=True, text=True)
    sg = os.path.join(CASES, name + ".sg")
    ok = os.path.exists(sg)
    out = os.path.join(DATA, "solid_rh%02d.yaml" % rh)
    nelem = "-"
    if ok:
        r2 = subprocess.run([PY, CONV, sg, out], capture_output=True, text=True)
        for ln in r2.stdout.splitlines():
            if "nelem=" in ln:
                nelem = ln.strip()
    print("rh=%2d  h=%.5f  oml=%.5f  ms=%.6f  sg=%s  %s"
          % (rh, h, oml, ms, "OK" if ok else "FAIL", nelem), flush=True)
print("done")
