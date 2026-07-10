"""2_generate_shell_mesh.py -- generate the 3-D SHELL (quad) tapered-segment SG and its
1-D shell boundary contours with opensg_io, from a windIO blade.

Mid-surface reference = OML (coincides with the solid's outer hex ring), span-
interpolated per-bay layup, NuMAD material orientation.  Writes, into <outdir>:
shell_segment.yaml,  shell_boundary_L.yaml,  shell_boundary_R.yaml.

Run (server opensg_2_0):
    python examples/taper/2_generate_shell_mesh.py <r1> <r2> <outdir> [windio.yaml]
"""
import os
import sys

import yaml

from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import (hex_between_sections, shell_between_sections,
                                shell_yaml_payload, shell_boundary_payload,
                                assert_shell_conforming)

NR, NSP, NW, MESH = 4, 12, 3, 0.02


def main(r1, r2, outdir, windio):
    os.makedirs(outdir, exist_ok=True)
    blade = load_blade(windio)
    cs1 = build_cross_section(blade, r1, mesh_size=MESH)
    cs2 = build_cross_section(blade, r2, mesh_size=MESH)
    z1, z2 = blade_span_z(blade, r1), blade_span_z(blade, r2)
    print("SHELL mesh  r=%.3f->%.3f  chord %.3f->%.3f  z=[%.2f, %.2f] m"
          % (r1, r2, cs1["chord"], cs2["chord"], z1, z2))

    res = hex_between_sections(cs1, cs2, z1, z2, nr=NR, nsp=NSP, nw=NW, mesh_size=MESH)
    shell = shell_between_sections(res, cs1, cs2, reference="OML")
    njunc = assert_shell_conforming(shell, len(cs1["webs"]), NSP)
    print("  SHELL %d nodes / %d quads ; branched conformity PASS (%d T-junctions, %d layups)"
          % (len(shell["nodes"]), len(shell["quads"]), njunc, len(shell["sections"])))

    yaml.safe_dump(shell_yaml_payload(shell, blade, _mat_block),
                   open(os.path.join(outdir, "shell_segment.yaml"), "w"),
                   default_flow_style=None, sort_keys=False)
    for si, tag in ((0, "L"), (1, "R")):
        yaml.safe_dump(shell_boundary_payload(res, shell, cs1, cs2, si, blade, _mat_block),
                       open(os.path.join(outdir, "shell_boundary_%s.yaml" % tag), "w"),
                       default_flow_style=None, sort_keys=False)
    print("  wrote shell_segment.yaml + shell_boundary_{L,R}.yaml  ->", outdir)


if __name__ == "__main__":
    r1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    r2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "out")
    windio = sys.argv[4] if len(sys.argv) > 4 else WINDIO
    main(r1, r2, outdir, windio)
