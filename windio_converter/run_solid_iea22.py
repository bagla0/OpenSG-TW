"""FEniCS-2D-solid Timoshenko 6x6 for the IEA-22 validation stations. Run in WSL opensg_env_v8."""
import os, sys, time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG); os.chdir(PKG)
os.environ.pop("C1_PENALTY", None)
VAL = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/windio_converter/validation"
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun

t0 = time.time()
for tag in (sys.argv[1:] or ["r030", "r050", "r070"]):
    yp = os.path.join(VAL, "solid_iea22_%s.yaml" % tag)
    if not os.path.exists(yp):
        print("skip (missing)", tag, flush=True); continue
    try:
        sm = SolidBounMesh(yp); mp, _ = sm.material_database
        Cs = np.asarray(compute_timo_boun(mp, sm.meshdata)[0]); Cs = 0.5 * (Cs + Cs.T)
        np.savetxt(os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag), Cs,
                   header="FEniCS-2D-solid IEA-22 %s [EA GA2 GA3 GJ EI2 EI3]" % tag)
        print("%s %5.0fs  EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e"
              % (tag, time.time() - t0, Cs[0, 0], Cs[1, 1], Cs[2, 2], Cs[3, 3], Cs[4, 4], Cs[5, 5]), flush=True)
    except Exception as e:                              # e.g. OOM on the thick root -> skip, keep going
        print("%s FAILED %5.0fs : %s" % (tag, time.time() - t0, repr(e)[:80]), flush=True)
print("DONE", flush=True)
