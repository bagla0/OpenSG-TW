"""Run the whole reproduction pipeline in order (steps 0 -> 4).

    python run_all.py

Equivalent to running, from this folder:
    python 0_generate_meshes.py
    python 1_run_kirchhoff.py
    python 2_run_rm.py
    python 3_orientation.py
    python 4_compare_to_solid.py
"""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
for step in ("0_generate_meshes.py", "1_run_kirchhoff.py", "2_run_rm.py",
             "3_orientation.py", "4_compare_to_solid.py"):
    print("\n" + "#" * 92 + "\n# %s\n" % step + "#" * 92)
    runpy.run_path(os.path.join(HERE, step), run_name="__main__")
print("\nALL STEPS DONE.")
