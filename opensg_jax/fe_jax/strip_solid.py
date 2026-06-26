"""
FEniCS 2D-solid (VABS-equivalent) cross-sectional analysis -> Timoshenko 6x6.

Reads the 2D solid cross-section strip_iso_solid.yaml and computes the
Timoshenko 6x6 with a full 3D constitutive law (the high-fidelity reference for
the RM/Kirchhoff shell models).  Order [ext, shear2, shear3, twist, bend2, bend3].

This is the SOLID counterpart of strip_RM.py / strip_Kirchhoff.py and must be run
in WSL with the FEniCSx environment (opensg_env_v8), e.g.:

  wsl bash -lc "source ~/miniconda3/etc/profile.d/conda.sh; conda activate opensg_env_v8; \
       python '/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/strip_solid.py'"
"""
import os, sys
import numpy as np

# opensg FEniCS package (training-data copy) on the path
PKG = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code/tests/research/training data/opensg-FEniCS"
sys.path.insert(0, PKG)
HERE = "/mnt/c/Users/bagla0/OneDrive - purdue.edu/2026_195/Claude_code"
YAML = os.path.join(HERE, "strip_iso_solid.yaml")
LBL = ["ext", "shear2", "shear3", "twist", "bend2", "bend3"]

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun


def solid_timoshenko_6x6(yaml_path):
    sm = SolidBounMesh(yaml_path)              # build the 2D solid mesh + material map
    mat_params, _density = sm.material_database
    C6 = np.asarray(compute_timo_boun(mat_params, sm.meshdata)[0])   # solve -> 6x6
    return C6


if __name__ == "__main__":
    print(f"reading 2D solid: {YAML}")
    C6 = solid_timoshenko_6x6(YAML)
    print(f"\n=== FEniCS 2D-solid (VABS) Timoshenko 6x6  order {LBL} ===")
    for i in range(6):
        print("  " + "".join(f"{C6[i, j]:14.4e}" for j in range(6)))
    d = np.diag(C6)
    print("  diagonal:  EA={:.4e}  GA2={:.4e}  GA3={:.4e}  GJ={:.4e}  EI2={:.4e}  EI3={:.4e}"
          .format(d[0], d[1], d[2], d[3], d[4], d[5]))
