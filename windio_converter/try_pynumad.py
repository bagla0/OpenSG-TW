"""Honest hands-on attempt to use pyNuMAD for a 2D-solid cross-section mesh from the IEA-22 windIO file."""
import os, windIO, traceback
import pynumad
from pynumad.objects.blade import Blade

src = os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml")

print("ATTEMPT 1: load the IEA-22 (windIO v2) directly into a pyNuMAD Blade")
try:
    bl = Blade()
    from pynumad.io.yaml_to_blade import yaml_to_blade
    yaml_to_blade(bl, src)
    print("  -> SUCCESS (unexpected)")
except Exception as e:
    print("  -> FAILED:", type(e).__name__, str(e)[:120])
    tb = traceback.format_exc().strip().splitlines()
    print("     at:", tb[-2].strip() if len(tb) > 1 else "?")

print("\nATTEMPT 2: is there a v2->v1 windIO converter to feed pyNuMAD?")
import windIO.converters.windIO2windIO as cv
print("  windIO2windIO classes:", [n for n in dir(cv) if n[0].isalpha() and n[0].islower() is False and "to" in n.lower()])

print("\nATTEMPT 3: pure-Python 2D mesher availability (headless, no Cubit)")
from pynumad.mesh_gen.mesh2d import Mesh2D
print("  Mesh2D importable:", Mesh2D is not None, "-> generic boundary->area mesher (single region)")
try:
    import cubit
    print("  cubit importable:", True)
except Exception as e:
    print("  cubit importable:", False, "(", type(e).__name__, ") -> pyNuMAD make_cross_sections path UNAVAILABLE")

print("\nATTEMPT 4: bundled pyNuMAD example blades (any v1 yaml to drive the mesher)?")
pdir = os.path.dirname(pynumad.__file__)
for root, dirs, files in os.walk(pdir):
    for f in files:
        if f.endswith((".yaml", ".yml")) and "__" not in root:
            print("  ", os.path.relpath(os.path.join(root, f), pdir))
