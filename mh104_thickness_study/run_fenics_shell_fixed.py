"""WSL: re-run the FEniCS-shell OML 6x6 with the now-default GENERALIZED penalty (BETA*max_D) for
f=0.1..0.75, and verify the iso tube is unchanged (penalty ~ old value there).  Overwrites
results/C6_fenics_shell_OML_f0NN.txt."""
import os
import sys
import time
import numpy as np

PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/training data/opensg-FEniCS"
sys.path.insert(0, PKG); os.chdir(PKG)
os.environ.pop("C1_PENALTY", None)                  # use the generalized BETA*max_D
from opensg.mesh.segment import ShellBounMesh

CC = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code"
STUDY = CC + "/mh104_thickness_study"; RES = STUDY + "/results"
t0 = time.time()

# --- tube validation (generalized penalty must reproduce the old behaviour: EI2~EI3, penalty~1e13) ---
for tube in (CC + "/benchmark_tube/data/shell_iso_0.06.yaml", CC + "/data/1Dshell_tube_iso_006.yaml"):
    if os.path.exists(tube):
        try:
            shm = ShellBounMesh(tube); ABD, _ = shm.compute_ABD(frac=0.5)
            mD = float(np.max(np.abs(np.asarray(ABD)[..., 3:, 3:])))
            C = np.asarray(shm.compute_timo(ABD)[1]); C = 0.5 * (C + C.T)
            print("TUBE %s: max_D=%.2e penalty(1e7*max_D)=%.2e  EI2=%.4e EI3=%.4e (ratio %.3f)  GA2=%.3e GA3=%.3e" % (
                os.path.basename(tube), mD, 1e7 * mD, C[4, 4], C[5, 5], C[4, 4] / C[5, 5], C[1, 1], C[2, 2]), flush=True)
            break
        except Exception as e:
            print("tube %s skipped: %s" % (os.path.basename(tube), e), flush=True)

# --- mh104 FEniCS-shell sweep with the generalized penalty ---
for fi in (10, 20, 30, 40, 60, 75):
    shy = STUDY + "/debug/shell_ref_f%03d_connect.yaml" % fi
    sp = RES + "/C6_solid_f%03d.txt" % fi
    so = np.loadtxt(sp) if os.path.exists(sp) else None
    shm = ShellBounMesh(shy); ABD, _ = shm.compute_ABD(frac=0.0)
    C = np.asarray(shm.compute_timo(ABD)[1]); C = 0.5 * (C + C.T)
    np.savetxt(RES + "/C6_fenics_shell_OML_f%03d.txt" % fi, C, header="FEniCS-shell OML generalized-penalty f=%.2f" % (fi / 100))
    msg = "f=%.2f (%3.0fs)  EA=%.3e GA2=%.3e GA3=%.3e GJ=%.3e EI2=%.3e EI3=%.3e" % (
        fi / 100, time.time() - t0, C[0, 0], C[1, 1], C[2, 2], C[3, 3], C[4, 4], C[5, 5])
    if so is not None:
        msg += "  | GA3=%+.0f%% EI2=%+.0f%%" % (100 * (C[2, 2] - so[2, 2]) / so[2, 2], 100 * (C[4, 4] - so[4, 4]) / so[4, 4])
    print(msg, flush=True)
print("done (%.0fs)" % (time.time() - t0), flush=True)
