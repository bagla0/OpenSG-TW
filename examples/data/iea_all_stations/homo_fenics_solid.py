'''
homo_fenics_solid.py  --  FEniCSx/dolfinx 2-D SOLID homogenization of the 2d_yaml meshes -> Timoshenko 6x6
==========================================================================================================
Runs the FEniCS (dolfinx) finite-element MSG solid cross-sectional solver on every 2-D solid SG YAML in
<dir> and writes the Timoshenko 6x6 stiffness (VABS order: EA, GA2, GA3, GJ, EI2, EI3) per station.

This is the FEniCSx counterpart of `homo_jax_solid.py` (JAX-FEM solid) and `homo_rm_shell.py` (RM shell):
it reads the SAME 2-D solid SG YAMLs that the JAX solver reads and returns the same Timoshenko 6x6, but
through the FEniCS `opensg` fork (dolfinx 0.8.0) instead of JAX.

    segment_mesh = SolidBounMesh(<name>_solid.yaml)                 # opensg.mesh.segment (FEniCS fork)
    mat_param, density = segment_mesh.material_database
    meshdata           = segment_mesh.meshdata                      # {mesh, subdomains, frame, origin}
    C6 = compute_timo_boun(mat_param, meshdata)[0]                  # opensg.core.solid  -> VABS-sorted 6x6

The call sequence mirrors OpenSG-1.0/examples/1_get_beam_props_from_solid_cross_section.py and
OpenSG-1.0/examples/test/run_solid_timo_from_yaml.py exactly (SolidBounMesh -> compute_timo_boun).
`compute_timo_boun` returns (Deff_srt, V0, V1s); Deff_srt is already re-sorted into VABS order so its
diagonal is [EA, GA2, GA3, GJ, EI2, EI3].  No YAML-format adaptation is needed: the OpenSG_io 2-D solid
YAML (nodes / elements / sets / elementOrientations / materials, node & element rows written as a single
space-separated string inside a 1-element list) is exactly the native schema `SolidBounMesh` parses.

Each station's mesh has a different number of nodes/elements (ragged), so stations run in a plain loop;
each SolidBounMesh(...) drops scratch `SG_mesh.msh/.xdmf/.h5` in the working dir, so this script cd's into
<out>/_scratch before building each mesh to keep the repo tree clean.

Environment (FEniCSx / dolfinx 0.8.0):
    ~/miniconda3/envs/opensg_env_v8/bin/python homo_fenics_solid.py                # all solids in 2d_yaml/
    ~/miniconda3/envs/opensg_env_v8/bin/python homo_fenics_solid.py --r 0.247      # one station
Output: <out>/C6_fenics_<name>.txt (the 6x6) + printed diagonal per station.
==========================================================================================================
'''
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def find_opensg_fenics():
    """Locate the FEniCS `opensg` fork (has opensg/core/solid.py + SolidBounMesh)."""
    cands = [
        os.path.expanduser("~/claude_tmp/opensg-FEniCS"),
        os.path.expanduser("~/claude_tmp/OpenSG-1.0"),
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "opensg", "core", "solid.py")):
            return c
    return None


PKG = find_opensg_fenics()
if PKG is None:
    sys.exit("could not find the FEniCS opensg fork (opensg-FEniCS/ or OpenSG-1.0/)")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import opensg  # noqa: E402  (side-effect: registers subpackages)
from opensg.mesh.segment import SolidBounMesh  # noqa: E402
from opensg.core.solid import compute_timo_boun  # noqa: E402

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def _reset_gmsh():
    """SolidBounMesh calls gmsh.initialize() without a matching finalize(); finalize any stale
    session so the next station gets a clean gmsh model (defensive, safe if gmsh not yet imported)."""
    try:
        import gmsh
        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass


def compute_c6(yaml_path):
    """SolidBounMesh -> compute_timo_boun -> VABS-sorted Timoshenko 6x6 (numpy)."""
    segment_mesh = SolidBounMesh(yaml_path)
    material_parameters, density = segment_mesh.material_database
    meshdata = segment_mesh.meshdata
    C6 = np.asarray(compute_timo_boun(material_parameters, meshdata)[0])
    return C6


def main():
    ap = argparse.ArgumentParser(description="FEniCSx 2-D solid Timoshenko 6x6 from 2-D solid YAMLs")
    ap.add_argument("--dir", default=os.path.join(HERE, "2d_yaml"))
    ap.add_argument("--r", type=float, default=None, help="single station r (else all in --dir)")
    ap.add_argument("--out", default=os.path.join(HERE, "homo_fenics"))
    ap.add_argument("--glob", default="*_solid.yaml")
    a = ap.parse_args()

    a.dir = os.path.abspath(a.dir)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.dir, a.glob)))
    files = [f for f in files if "t1only" not in os.path.basename(f)]     # skip the t1-only variant
    if a.r is not None:
        tg = "r%04d" % round(a.r * 1000)
        files = [f for f in files if tg in os.path.basename(f)]

    print("OpenSG-FEniCSx 2D solid -> Timoshenko 6x6; %d"
          % len(files), flush=True)

    # keep the SG_mesh.* scratch files out of the repo tree
    work = os.path.join(a.out, "_scratch")
    os.makedirs(work, exist_ok=True)
    cwd0 = os.getcwd()

    for f in files:
        f = os.path.abspath(f)
        nm = os.path.basename(f).replace("_solid.yaml", "").replace(".yaml", "")
        t0 = time.time()
        try:
            _reset_gmsh()
            os.chdir(work)
            C6 = compute_c6(f)
            os.chdir(cwd0)
            np.savetxt(os.path.join(a.out, "C6_fenics_%s.txt" % nm), C6)
            d = "  ".join("%s=%.4g" % (LBL[i], C6[i, i]) for i in range(6))
            print("[%-10s] %s  [%.1fs]" % (nm, d, time.time() - t0), flush=True)
        except Exception as e:
            os.chdir(cwd0)
            print("[%-10s] FAIL %s" % (nm, repr(e)[:170]), flush=True)

    print("\nwrote OpenSG_Fenicsx_*.txt ->", a.out)


if __name__ == "__main__":
    main()
