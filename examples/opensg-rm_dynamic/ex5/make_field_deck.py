"""make_field_deck.py -- the FULL-FIELD snapshot deck for the 3-D rendered
OpenSG-RM field: the standard Ex.5 RM step deck plus whole-plate prints
(all 441 nodal U + all 400-element SF/SM) every 57th increment -- 57 x 50us
= 2.85 ms, the first response peak, so one dump lands exactly there.
rm_field_vtk.py consumes the resulting sandwich_RM_field.dat.

Run:  python examples/opensg-rm_dynamic/ex5/make_field_deck.py
then on the Abaqus machine:  abaqus job=sandwich_RM_field cpus=4 interactive
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_abaqus_dyn as m

path = m.write_rm(os.path.join(HERE, "sandwich_RM_field.inp"), "step")
txt = open(path).read()
extra = ("*NODE PRINT, NSET=NALL, FREQUENCY=57\n"
         "U\n"
         "*EL PRINT, ELSET=EALL, FREQUENCY=57\n"
         "SF, SM\n")
txt = txt.replace("*END STEP", extra + "*END STEP")
open(path, "w").write(txt)
print("wrote %s (full-field prints every 57 increments = 2.85 ms)"
      % os.path.basename(path))
