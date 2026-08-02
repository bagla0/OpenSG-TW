"""make_field_deck_ex2.py -- the Ex.2 FULL-FIELD snapshot decks, mirroring
the ex5 pipeline on the (0/90/0) a=5h plate, step pulse:

  ex2_RM_field.inp     the RM step deck + whole-plate prints (all 441
                       nodal U + all 400-element SF/SM) every 15th
                       increment -- 15 x 50 us = 0.75 ms, the FIRST
                       RESPONSE PEAK, so dump #1 is the comparison instant
  ex2_SOLID_field.inp  the 3-D benchmark rerun with *DYNAMIC,DIRECT at
                       dt = 50 us and field frames every 15 increments,
                       so its frame sits at the IDENTICAL instant

Run:  python examples/opensg-rm_dynamic/ex2/make_field_deck_ex2.py
then on the Abaqus machine (claude_tmp/pending2.bat runs everything).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_abaqus_ex2 as m

# ---- RM whole-plate snapshot deck --------------------------------------
path = m.write_rm("step")
txt = open(path).read()
extra = ("*NODE PRINT, NSET=NALL, FREQUENCY=15\n"
         "U\n"
         "*EL PRINT, ELSET=EALL, FREQUENCY=15\n"
         "SF, SM\n")
txt = txt.replace("*END STEP", extra + "*END STEP")
new = os.path.join(HERE, "ex2_RM_field.inp")
open(new, "w").write(txt)
print("wrote %s (full-field prints every 15 inc = 0.75 ms, the peak)"
      % os.path.basename(new))

# ---- solid snapshot deck: DIRECT fixed dt + matched field frames -------
spath = m.write_solid("step")
txt = open(spath).read()
txt = txt.replace("*DYNAMIC\n5e-05, 0.02, 5e-09, 5e-05",
                  "*DYNAMIC,DIRECT\n5e-05, 0.02")
txt = txt.replace("*OUTPUT, FIELD, FREQUENCY=50",
                  "*OUTPUT, FIELD, FREQUENCY=15")
new = os.path.join(HERE, "ex2_SOLID_field.inp")
open(new, "w").write(txt)
print("wrote %s (DIRECT dt=50us, field frames every 15 inc = 0.75 ms)"
      % os.path.basename(new))
