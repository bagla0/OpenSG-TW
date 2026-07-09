"""_build_nb.py -- build + EXECUTE the 3 RM_taper tutorial notebooks (circle, square,
webbed ellipse). Each notebook renders the shell mesh + e2/e3 orientation, then runs the
6-DOF RM shell for the boundary ring AND the tapered segment (thin + thick) and prints the
Timoshenko 6x6 with its %-error vs the conforming 3-D solid. Run on the server (compute env);
needs nbformat + nbconvert + ipykernel."""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

CC = os.path.expanduser("~/OpenSG-TW-claude")
TUT = os.path.join(CC, "docs", "tutorials")

GEOM = {
    "circle": dict(title="circular tube", mesh="taper_study", tgfmt="{r}_m45_aR070",
                   ref="taper_study_solid_m45.npz",
                   blurb="A smoothly curved single-cell tube, taper ratio $R_R/R_L=0.7$."),
    "square": dict(title="square tube", mesh="taper_square", tgfmt="{r}_m45_aR070",
                   ref="taper_square_solid_m45.npz",
                   blurb="A flat-walled tube ($k_{22}=0$ on the faces) -- the case that "
                         "exposes the drilling degeneracy MITC would alias."),
    "ellipse": dict(title="webbed ellipse", mesh="rm_taper_ellipse", tgfmt="{r}_m45",
                    ref="ellipse_solid_m45.npz",
                    blurb="A blade-like multi-cell ellipse with three internal shear webs "
                          "-- the most demanding case in the paper."),
}

INTRO = """# Tapered {title} -- RM shell vs 3-D solid

Homogenizes the equivalent-beam **Timoshenko $6\\times6$** ($C^b$) of the tapered {title}
with a single **6-DOF independent-$\\omega_3$ Reissner--Mindlin shell**, for both the
**boundary ring** and the **tapered segment**, at thin ($t/R=0.02$) and thick ($t/R=0.20$)
walls, single $[-45^\\circ]$ ply, and compares against a conforming 3-D FEniCS solid.

{blurb}

Transverse-shear scheme (6-DOF everywhere): the tapered segment uses **full integration** at
every thickness (locking-free; MITC would alias the drilling on flat walls); the boundary ring
uses a $\\gamma_{{23}}$-tie on the thin wall and full integration on the thick wall.

This notebook runs `examples/RM_taper/{geom}.py` inline."""

SETUP = r'''%matplotlib inline
import os, sys, numpy as np
def _root(d):
    d = os.path.abspath(d)
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")): return d
        d = os.path.dirname(d)
    return os.getcwd()
CC = _root(os.getcwd())
sys.path.insert(0, os.path.join(CC, "examples", "RM_taper"))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
import _rm_common as rm
np.set_printoptions(precision=4, suppress=True, linewidth=140)
MESH = os.path.join(CC, "examples", "data", "MESHNAME", "meshes")
REF  = np.load(os.path.join(CC, "examples", "data", "benchmark", "REFNAME"))
RES  = os.path.join(CC, "docs", "tutorials", "_rmout"); os.makedirs(RES, exist_ok=True)'''

ORIENT = r'''from IPython.display import Image, display
for r in ("thin", "thick"):
    tg = "TGFMT".replace("{r}", r)
    ttl = "GEOM -- %s wall (t/R=%s):  e2 blue, e3 black, webs crimson" % (r, "0.02" if r == "thin" else "0.20")
    png = rm.render_orientation(MESH, tg, RES, title=ttl)
    display(Image(filename=png))'''

SOLVE = r'''def show(label, So, Sh, shear):
    So = 0.5 * (So + So.T); Sh = 0.5 * (Sh + Sh.T)
    print("\n" + label + "   [6-DOF RM shell, shear=%s]" % shear)
    print("RM shell Timoshenko 6x6  (x1e9):")
    print(Sh / 1e9)
    e = 100 * (np.diag(Sh) - np.diag(So)) / np.diag(So)
    print("diagonal %err vs 3-D solid [EA, GA2, GA3, GJ, EI2, EI3]:", np.round(e, 1))

for r, tR in (("thin", 0.02), ("thick", 0.20)):
    tg = "TGFMT".replace("{r}", r)
    print("=" * 72)
    print("GEOM -- %s wall (t/R = %.2f)" % (r.upper(), tR))
    sb = rm.shear_for("boundary", tR); st = rm.shear_for("taper", tR)
    Cb = rm.solve_boundary(MESH, tg, RES, sb); show("BOUNDARY ring", REF[tg + "_L"], Cb, sb)
    Ct = rm.solve_taper(MESH, tg, RES, st);    show("TAPERED segment", REF[tg + "_seg"], Ct, st)'''

for geom, g in GEOM.items():
    setup = SETUP.replace("MESHNAME", g["mesh"]).replace("REFNAME", g["ref"])
    orient = ORIENT.replace("TGFMT", g["tgfmt"]).replace("GEOM", g["title"])
    solve = SOLVE.replace("TGFMT", g["tgfmt"]).replace("GEOM", g["title"])
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(INTRO.format(title=g["title"], blurb=g["blurb"], geom=geom)),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## Mesh and material orientation"),
        nbf.v4.new_code_cell(orient),
        nbf.v4.new_markdown_cell("## Timoshenko $6\\times6$ -- boundary ring and tapered segment"),
        nbf.v4.new_code_cell(solve),
    ]
    print("executing", geom, "...", flush=True)
    ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": CC}})
    out = os.path.join(TUT, "rm_taper_%s.ipynb" % geom)
    nbf.write(nb, out)
    print("wrote", out, flush=True)
print("done", flush=True)
