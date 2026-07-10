"""1_generate_solid_mesh.py -- generate the 3-D SOLID (hex) tapered-segment SG and its
2-D solid boundary cross-sections with opensg_io, from a windIO blade.

Ply-conforming through-thickness hex layers (sandwich skins meshed exactly), NuMAD
material orientation (e1=span, e3=inward), span-interpolated layup.  Writes, into
<outdir>:  solid_segment.yaml,  solid_boundary_L.yaml,  solid_boundary_R.yaml.

Run (server opensg_2_0):
    python examples/taper/1_generate_solid_mesh.py <r1> <r2> <outdir> [windio.yaml]
"""
import os
import sys

import numpy as np
import yaml

from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload, solid_boundary_payload
from opensg_io.mesh3d import export_solid_yaml
from opensg_io.conformity import assert_conforming, min_scaled_jacobian

NR, NSP, NW, MESH = 4, 12, 3, 0.02


def main(r1, r2, outdir, windio):
    os.makedirs(outdir, exist_ok=True)
    blade = load_blade(windio)
    cs1 = build_cross_section(blade, r1, mesh_size=MESH)
    cs2 = build_cross_section(blade, r2, mesh_size=MESH)
    z1, z2 = blade_span_z(blade, r1), blade_span_z(blade, r2)
    print("SOLID mesh  r=%.3f->%.3f  chord %.3f->%.3f  z=[%.2f, %.2f] m"
          % (r1, r2, cs1["chord"], cs2["chord"], z1, z2))

    res = hex_between_sections(cs1, cs2, z1, z2, nr=NR, nsp=NSP, nw=NW, mesh_size=MESH)
    nodes, hexes = res["nodes"], res["hexes"]
    assert_conforming(nodes, hexes, "hex")
    msj, ninv = min_scaled_jacobian(nodes, hexes)
    assert ninv == 0, "%d inverted hexes" % ninv
    print("  HEX %d nodes / %d hexes ; conformity PASS ; min scaled Jacobian %.3f"
          % (len(nodes), len(hexes), msj))

    oris, hmats = solid_yaml_payload(res, cs1, cs2)
    mat_names = sorted(set(hmats))
    sets = {"element": [{"name": m, "labels": [k + 1 for k, hm in enumerate(hmats) if hm == m]}
                        for m in mat_names]}
    mats = [{"name": m, **{k: _mat_block(blade, m)["elastic"][k] for k in ("E", "G", "nu")},
             "rho": _mat_block(blade, m)["density"]} for m in mat_names]
    export_solid_yaml(os.path.join(outdir, "solid_segment.yaml"), nodes, hexes, "hex", oris, mats, sets=sets)

    for si, tag in ((0, "L"), (1, "R")):
        yaml.safe_dump(solid_boundary_payload(res, cs1, cs2, si, blade, _mat_block),
                       open(os.path.join(outdir, "solid_boundary_%s.yaml" % tag), "w"),
                       default_flow_style=None, sort_keys=False)
    print("  wrote solid_segment.yaml + solid_boundary_{L,R}.yaml  ->", outdir)


if __name__ == "__main__":
    r1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    r2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "out")
    windio = sys.argv[4] if len(sys.argv) > 4 else WINDIO
    main(r1, r2, outdir, windio)
