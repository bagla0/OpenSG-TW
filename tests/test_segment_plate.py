"""Validation for the through-thickness plate SG generator
(``opensg_jax.fe_jax.segment_plate``): layup dict -> 1-D SG YAML -> 8x8 ABDG.

The headline case is a SYMMETRIC laminate, because symmetry is a property the round trip
cannot fake: a layup that is mirror-symmetric about its mid-surface has NO membrane-bending
coupling, so the B block of the 8x8 must vanish identically when the SG is referenced at
fraction = 0.5.  If the generated mesh got the ply order, a ply thickness, an angle, or the
reference plane wrong, B would light up.

Also covered: the written mesh is the SAME mesh msg_rm_plate builds internally, the YAML
round trip returns the layup it was given, and the file-driven 8x8 equals the direct call.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
import jax

jax.config.update("jax_enable_x64", True)

from opensg_jax.fe_jax.msg_rm_plate import _node_grid, rm_plate_msg
from opensg_jax.fe_jax.msg_transverse_shear import plate_8x8
from opensg_jax.fe_jax.segment_plate import (plate_sg_dict, plate_sg_mesh, plate_sg_yaml,
                                             read_plate_sg_yaml, rm_plate_from_yaml)

MDB = {"gr": {"E": [172.4e9, 6.89e9, 6.89e9], "G": [3.45e9, 1.38e9, 3.45e9],
              "nu": [0.25] * 3, "rho": 1600.0},
       "glass": {"E": [45e9, 12e9, 12e9], "G": [5.5e9, 5.5e9, 4.1e9],
                 "nu": [0.28, 0.28, 0.3], "rho": 1950.0}}

# mirror-symmetric about the mid-surface, in material AND angle AND thickness
SYM = {"mat_names": ["gr", "gr", "gr", "gr"], "thick": [0.002, 0.003, 0.003, 0.002],
       "angles": [45.0, -45.0, -45.0, 45.0]}
SYM3 = {"mat_names": ["glass", "gr", "glass"], "thick": [0.004, 0.010, 0.004],
        "angles": [0.0, 90.0, 0.0]}
ASYM = {"mat_names": ["gr", "gr"], "thick": [0.005, 0.005], "angles": [0.0, 90.0]}


def tmp_yaml(name):
    return os.path.join(tempfile.mkdtemp(prefix="platesg_"), name)


def test_mesh_matches_the_solver_mesh():
    """The mesh written to the YAML is exactly the one msg_rm_plate builds internally."""
    for lay in (SYM, SYM3, ASYM):
        for npl, p, frac in ((1, 4, 0.5), (2, 4, 0.0), (4, 3, 1.0)):
            node_x, elements, elem_ply = plate_sg_mesh(lay["thick"], npl, p, frac)
            ref = _node_grid([float(t) for t in lay["thick"]], npl, p,
                             frac * sum(float(t) for t in lay["thick"]))
            assert np.allclose(node_x, ref, atol=0, rtol=0)
            n_elem = len(lay["thick"]) * npl
            assert elements.shape == (n_elem, p + 1)
            assert node_x.size == p * n_elem + 1
            # elements tile the thickness with shared end nodes, no gaps or overlaps
            assert np.array_equal(elements[:-1, -1], elements[1:, 0])
            assert elem_ply.size == n_elem and elem_ply.max() == len(lay["thick"]) - 1


def test_reference_plane_places_the_nodes():
    h = sum(SYM["thick"])
    for frac in (0.0, 0.25, 0.5, 1.0):
        node_x, _, _ = plate_sg_mesh(SYM["thick"], fraction=frac)
        assert abs(node_x[0] - (-frac * h)) < 1e-15
        assert abs(node_x[-1] - (1.0 - frac) * h) < 1e-15


def test_yaml_round_trip_returns_the_layup():
    path = tmp_yaml("sym.yaml")
    plate_sg_yaml(path, SYM, MDB, fraction=0.5)
    got = read_plate_sg_yaml(path)
    assert got["mat_names"] == SYM["mat_names"]
    assert np.allclose(got["thick"], SYM["thick"])
    assert np.allclose(got["angles"], SYM["angles"])
    assert got["elem_order"] == 4 and got["n_per_layer"] == 1
    assert abs(got["fraction"] - 0.5) < 1e-12
    for m in ("gr",):
        assert np.allclose(got["material_db"][m]["E"], MDB[m]["E"])
        assert np.allclose(got["material_db"][m]["nu"], MDB[m]["nu"])


def test_five_noded_quartic_is_the_default():
    doc = plate_sg_dict(SYM, MDB)
    assert doc["sg"]["elem_order"] == 4
    assert all(len(el) == 5 for el in doc["elements"]), "elements must be 5-noded"
    assert len(doc["elements"]) == len(SYM["thick"])          # one element per ply
    assert len(doc["nodes"]) == 4 * len(SYM["thick"]) + 1
    assert [s["elementSet"] for s in doc["sections"]] == ["ply_%d" % (k + 1)
                                                         for k in range(len(SYM["thick"]))]


def test_symmetric_laminate_has_zero_B():
    """THE test: a mirror-symmetric layup referenced at its mid-surface has no
    membrane-bending coupling.  B must be zero relative to A and D."""
    for lay, tag in ((SYM, "[45/-45]s"), (SYM3, "[glass0/gr90/glass0]")):
        path = tmp_yaml("sym.yaml")
        plate_sg_yaml(path, lay, MDB, fraction=0.5)
        r = rm_plate_from_yaml(path)
        A, B, D = r["ABDG"][:3, :3], r["ABDG"][:3, 3:6], r["ABDG"][3:6, 3:6]
        rel = np.max(np.abs(B)) / np.sqrt(np.max(np.abs(A)) * np.max(np.abs(D)))
        assert rel < 1e-12, "%s: B/sqrt(A D) = %.2e, expected 0" % (tag, rel)


def test_asymmetric_laminate_has_nonzero_B():
    """The converse, so the test above is not vacuous: [0/90] IS coupled."""
    path = tmp_yaml("asym.yaml")
    plate_sg_yaml(path, ASYM, MDB, fraction=0.5)
    r = rm_plate_from_yaml(path)
    A, B, D = r["ABDG"][:3, :3], r["ABDG"][:3, 3:6], r["ABDG"][3:6, 3:6]
    rel = np.max(np.abs(B)) / np.sqrt(np.max(np.abs(A)) * np.max(np.abs(D)))
    assert rel > 1e-2, "[0/90] should be strongly coupled, got B/sqrt(A D) = %.2e" % rel


def test_yaml_driven_equals_direct_call():
    """rm_plate_from_yaml must reproduce rm_plate_msg on the same layup, bit for bit."""
    for lay in (SYM, SYM3, ASYM):
        path = tmp_yaml("case.yaml")
        plate_sg_yaml(path, lay, MDB, fraction=0.5)
        rf = rm_plate_from_yaml(path)
        rd = rm_plate_msg(lay["thick"], lay["angles"], lay["mat_names"], MDB, fraction=0.5)
        ref = np.asarray(plate_8x8(np.asarray(rd["A6"]), np.asarray(rd["G_msg"])))
        assert np.max(np.abs(rf["ABDG"] - ref)) == 0.0


def test_corrupt_file_is_caught():
    """The reader trusts the stored ply thickness / reference fraction for exactness, so
    it must verify them against the mesh -- a file whose header and nodes disagree fails."""
    import yaml as _yaml

    path = tmp_yaml("sym.yaml")
    plate_sg_yaml(path, SYM, MDB, fraction=0.5)
    good = _yaml.safe_load(open(path))

    bad = {**good, "sections": [dict(s) for s in good["sections"]]}
    bad["sections"][0]["thickness"] = 0.5 * bad["sections"][0]["thickness"]
    _yaml.safe_dump(bad, open(path, "w"), sort_keys=False)
    with pytest.raises(ValueError, match="span"):
        read_plate_sg_yaml(path)

    bad2 = {**good, "sg": {**good["sg"], "reference_fraction": 0.0}}
    _yaml.safe_dump(bad2, open(path, "w"), sort_keys=False)
    with pytest.raises(ValueError, match="reference_fraction"):
        read_plate_sg_yaml(path)


def test_bad_layup_is_rejected():
    with pytest.raises(ValueError):
        plate_sg_dict({"mat_names": ["gr"], "thick": [0.1, 0.2], "angles": [0.0]}, MDB)
    with pytest.raises(ValueError):
        plate_sg_dict({"mat_names": ["gr"], "thick": [0.0], "angles": [0.0]}, MDB)
    with pytest.raises(KeyError):
        plate_sg_dict({"mat_names": ["nope"], "thick": [0.1], "angles": [0.0]}, MDB)


if __name__ == "__main__":
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  %s" % nm)
            except AssertionError as exc:
                fails += 1
                print("FAIL  %s: %s" % (nm, exc))
    print("\n%s" % ("all tests passed" if not fails else "%d FAILURES" % fails))
    sys.exit(1 if fails else 0)
