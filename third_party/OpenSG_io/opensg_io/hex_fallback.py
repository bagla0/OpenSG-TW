"""opensg_io.hex_fallback -- structured-HEX fallback for DEGENERATE cross-sections that
PreVABS / gmsh cannot triangulate (blade tip r=1.0, root-transition r~0.136: edge
recovery fails or the mesher hangs even at tiny mesh_size).

`to_solid_hex` builds the OpenSG_io structured-hex loft of a ONE-station prismatic segment
(pure numpy, no gmsh) and writes its z=0 boundary cross-section as a 2-D solid OpenSG YAML.
The output is the SAME schema the PreVABS `.sg -> convert_sg_to_yaml` route produces
(string/1-based "x y z" nodes, quad elements, per-material element sets, 9-float
elementOrientations [e1,e2,e3], flat E/G/nu/rho materials) so it is read unchanged by the
JAX homogenizer (opensg_jax read_solid_yaml / compute_timo_from_yaml) and the FEniCS one
(opensg SolidBounMesh / compute_timo_boun).
"""
import numpy as np
import yaml

from .converter import build_cross_section, _mat_block
from .hex_loft import hex_between_sections, solid_boundary_payload


class _Flow(list):
    pass


yaml.add_representer(_Flow, lambda d, x: d.represent_sequence(
    "tag:yaml.org,2002:seq", x, flow_style=True))


def _flow_doc(pay):
    """Wrap the payload's row lists in flow-style lists so the file matches the existing
    PreVABS-produced 2d_yaml exactly (cosmetic; the readers are style-agnostic)."""
    return {
        "nodes": [_Flow(n) for n in pay["nodes"]],
        "elements": [_Flow(e) for e in pay["elements"]],
        "sets": {"element": [{"name": s["name"], "labels": _Flow(list(s["labels"]))}
                             for s in pay["sets"]["element"]]},
        "elementOrientations": [_Flow([float(v) for v in o]) for o in pay["elementOrientations"]],
        "materials": [{"name": m["name"], "E": _Flow(list(m["E"])), "G": _Flow(list(m["G"])),
                       "nu": _Flow(list(m["nu"])), "rho": float(m["rho"])} for m in pay["materials"]],
    }


def to_solid_hex(blade, r, out_yaml, mesh_size=0.02, nr=4, nw=3):
    """windIO `blade` + span station `r` in [0,1]  ->  2-D solid cross-section YAML at
    `out_yaml`, built by the structured-hex loft (the gmsh-free fallback).

    nr = through-thickness hex layers, nw = web-junction band subdivisions, mesh_size =
    chord-normalised in-plane arc size (matches build_cross_section / hex_between_sections).
    Returns dict(chord, n_webs, n_nodes, n_quads, mats)."""
    cs = build_cross_section(blade, r, mesh_size=mesh_size)
    res = hex_between_sections(cs, cs, 0.0, 1.0, nr=nr, nsp=1, nw=nw, mesh_size=mesh_size)
    if res["n_still_inverted"] > 0:
        raise RuntimeError("hex loft left %d inverted cells at r=%.4f (section too degenerate)"
                           % (res["n_still_inverted"], r))
    pay = solid_boundary_payload(res, cs, cs, 0, blade, _mat_block)     # si=0 -> z=0 face
    with open(out_yaml, "w") as f:
        yaml.dump(_flow_doc(pay), f, sort_keys=False, default_flow_style=False)
    return dict(chord=float(cs["chord"]), n_webs=len(cs["webs"]), n_nodes=len(pay["nodes"]),
                n_quads=len(pay["elements"]), mats=[m["name"] for m in pay["materials"]])
