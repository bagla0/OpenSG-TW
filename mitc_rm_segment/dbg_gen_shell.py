"""dbg_gen_shell.py -- run the KNOWN-GOOD gen_ell3w webbed tapered shell through
shell_solve_lagrange (full) as a reference, and print sample skin/web orientation
frames so they can be compared to the PreVABS loft's frames."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
SP = sys.argv[1]
import yaml

import run_ell3w as e3w
e3w.T = 0.2
mdir = os.path.join(SP, "gen_ell3w_thick")
e3w.gen_ell3w(mdir, nc=48, nl=12, nw=6, nr=4)

import segment_indep
orig = segment_indep.assemble_segment_indep
def patched(*a, **k):
    k["shear"] = "full"; return orig(*a, **k)
segment_indep.assemble_segment_indep = patched
import importlib, run_indep as ri
importlib.reload(ri)
Sh = np.asarray(ri.shell_solve_lagrange("e3w", mdir, os.path.join(mdir, "res")))
segment_indep.assemble_segment_indep = orig
print("gen_ell3w thick shell (full) diag x1e9:", np.round(np.diag(Sh) / 1e9, 4))

d = yaml.safe_load(open(os.path.join(mdir, "shell_e3w.yaml")))
ori = np.array(d["elementOrientations"]); nq = len(d["elements"])
# gen_ell3w: nc*nl skin quads first, then web quads
nskin = 48 * 12
print("gen skin  ori[0]:   e1=%s e2=%s e3=%s" % (np.round(ori[0, :3], 3), np.round(ori[0, 3:6], 3), np.round(ori[0, 6:9], 3)))
print("gen web   ori[%d]: e1=%s e2=%s e3=%s" % (nskin + 5, np.round(ori[nskin + 5, :3], 3), np.round(ori[nskin + 5, 3:6], 3), np.round(ori[nskin + 5, 6:9], 3)))

# my loft frames
d2 = yaml.safe_load(open(os.path.join(SP, "segsh_ell_thick", "shell_ellTseg.yaml")))
o2 = np.array(d2["elementOrientations"])
sets = {g["name"]: set(g["labels"]) for g in d2["sets"]["element"]}
skin_labels = sets.get("layup_0", set()); web_labels = sets.get("layup_1", set())
si = min(skin_labels) - 1; wi = min(web_labels) - 1
print("loft skin ori[%d]:   e1=%s e2=%s e3=%s" % (si, np.round(o2[si, :3], 3), np.round(o2[si, 3:6], 3), np.round(o2[si, 6:9], 3)))
print("loft web  ori[%d]:  e1=%s e2=%s e3=%s" % (wi, np.round(o2[wi, :3], 3), np.round(o2[wi, 3:6], 3), np.round(o2[wi, 6:9], 3)))
# web layup angle check
print("gen sections:", [(s.get("elementSet"), s.get("layup")) for s in d["sections"]])
print("loft sections:", [(s.get("elementSet"), s.get("layup")) for s in d2["sections"]])
