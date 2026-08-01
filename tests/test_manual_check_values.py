"""The manual/tutorial CHECK VALUES must match the code.

docs/rm_plate_manual/gen_check_values.py freezes the numbers the user manual
and tutorials quote (8x8 anchors of the four tutorial laminates + the
tutorial-1 recovery anchors) into check_values.json.  This test recomputes
them: if a code change moves any number, it fails until the generator is
re-run and the documents rebuilt -- the docs can never silently drift.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.dirname(HERE)
sys.path.insert(0, CC)
sys.path.insert(0, os.path.join(CC, "docs", "rm_plate_manual"))

import gen_check_values as g                                       # noqa: E402

JSON = os.path.join(CC, "docs", "rm_plate_manual", "check_values.json")


def test_check_values_match_code():
    with open(JSON) as f:
        frozen = json.load(f)
    fresh = g.compute()
    for case, vals in fresh.items():
        for key, v in vals.items():
            if isinstance(v, str):
                continue
            ref = frozen[case][key]
            assert np.isclose(v, ref, rtol=1e-10), (
                "check value %s/%s drifted: docs say %r, code gives %r -- "
                "re-run docs/rm_plate_manual/gen_check_values.py and rebuild "
                "the manual" % (case, key, ref, v))
