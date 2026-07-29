"""Build docs/tutorials/plate_rm_8x8.ipynb -- a STANDALONE tutorial that runs entirely
from the repository's committed data (examples/data/...), so an external user can clone
the repo and execute it top to bottom.  No local/absolute paths.
    python docs/tutorials/_build_platerm_nb.py
then execute in place:
    jupyter nbconvert --to notebook --execute --inplace docs/tutorials/plate_rm_8x8.ipynb
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plate_rm_8x8.ipynb")
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# MSG-RM plate homogenization: the 8×8 ABDG wall law from a 1-D shell YAML

Every wall laminate of a thin-walled cross-section carries an **RM plate constitutive law**

$$\mathrm{ABDG} \;=\; \begin{bmatrix} A & B & 0\\ B & D & 0\\ 0 & 0 & G \end{bmatrix}\;(8\times8),$$

rows 1–6 the classical membrane/bending ABD (plate strains $[\epsilon_{11},\epsilon_{22},\gamma_{12},
\kappa_{11},\kappa_{22},\kappa_{12}]$), rows 7–8 the transverse shear $[2\gamma_{13},2\gamma_{23}]$.
This tutorial computes it for **every laminate of a 1-D shell SG YAML** with the **MSG/VAM
construction** of Yu, Hodges & Volovoi (*Computers & Structures* 81:439–454, 2003 — the plate twin
of the CMAME 2002 shell paper), implemented in core OpenSG as
`opensg_jax.fe_jax.msg_rm_plate.rm_plate_msg`:

| step | paper | code |
|------|-------|------|
| zeroth-order warping $V_0$ → classical ABD | Eq. (39)–(40) | `V0`, `A6` |
| first-order gradient warping $V_{11},V_{12}$ | Eq. (42)–(45) | `C1bar`, `C2bar` |
| second-order gradient energy $B,C,D$ | Eq. (46)–(47) | `H` (12×12) |
| RM projection: least squares of the residual $U^*$ over $X=G^{-1}$ **and Yu's 24 in-plane relaxed constants** (78 equations, 27 unknowns) | Eq. (55)–(60) | `blocks()`, `lstsq` → `G_msg`, `Ustar_rel` |

`Ustar_rel` is the fraction of the second-order gradient energy the RM functional could **not**
absorb — the distance from asymptotic correctness (0 means an exactly asymptotically-correct RM
model exists for that laminate).

**Input (committed in this repo):** `examples/data/1d_yaml/st15_shell.yaml` — the BAR-URC
station-15 blade cross-section (10 wall laminates, glass triax/UD + foam sandwich walls).
Command-line counterpart: `examples/6_get_plateRM_homo_using_1DSG.py`.
""")

code(r"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.abspath(os.getcwd())          # also runs from the repo root
sys.path.insert(0, ROOT)
np.set_printoptions(precision=4, linewidth=150)

from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg
from opensg_jax.fe_jax.msg_transverse_shear import plate_8x8, transverse_shear_stiffness

SHELL = os.path.join(ROOT, "examples", "data", "1d_yaml", "st15_shell.yaml")
nodes, elems, mdb, layup_db, elem_to_layup = load_yaml(SHELL)
print("%d wall laminates, %d contour elements, materials: %s"
      % (len(layup_db), len(elems), ", ".join(mdb)))
""")

md(r"""
## Sanity check first: homogeneous isotropic plate

For an isotropic single layer with $\nu=0$ the construction must return the textbook
$G = \tfrac{5}{6}\,G\,h$ **exactly**, with $U^*$ driven to machine zero (an asymptotically
correct RM model exists for this case).
""")

code(r"""
h = 0.01
mdb_iso = {"iso": {"E": [70e9]*3, "G": [35e9]*3, "nu": [0.0]*3, "rho": 1.0}}
r = rm_plate_msg([h], [0.0], ["iso"], mdb_iso, z_ref=h/2)
print("G_msg/(G*h) =", np.diag(r["G_msg"]) / (35e9*h), "  target 5/6 =", 5/6)
print("Ustar_rel   = %.2e" % r["Ustar_rel"])
""")

md(r"""
## The 8×8 ABDG for one wall laminate

Full matrix for the first laminate (center / mid-surface reference, `z_ref = h/2` — the
convention of the RM ring homogenization).
""")

code(r"""
ln = "layup_0"; lay = layup_db[ln]
thk = [float(t) for t in lay["thick"]]; ang = [float(a) for a in lay["angles"]]
mats = [str(m) for m in lay["mat_names"]]
h = float(sum(thk))
print(ln, ":", ", ".join("%s(%.1fmm/%g)" % (m, 1e3*t, a) for m, t, a in zip(mats, thk, ang)))

r = rm_plate_msg(thk, ang, mats, mdb, z_ref=0.5*h)
P8 = plate_8x8(r["A6"], r["G_msg"])
print("\nRM 8x8 ABDG  [[A,B,0],[B,D,0],[0,0,G]]:")
print(P8)
""")

md(r"""
## All wall laminates: MSG G vs Whitney, and the $U^*$ residual

The complementary-energy (Whitney) shear stiffness is shown for comparison — on sandwich
walls (soft foam core) the MSG least-squares $G$ comes out substantially **softer** than
Whitney, because the second-order energy feels the full core shear compliance.
""")

code(r"""
print("%-9s %2s %8s | %11s %11s | %11s %11s | %9s" %
      ("laminate", "np", "h [m]", "G11_msg", "G22_msg", "G11_Whit", "G22_Whit", "Ustar_rel"))
for ln, lay in layup_db.items():
    thk = [float(t) for t in lay["thick"]]; ang = [float(a) for a in lay["angles"]]
    mats = [str(m) for m in lay["mat_names"]]
    h = float(sum(thk))
    r = rm_plate_msg(thk, ang, mats, mdb, z_ref=0.5*h)
    Gw = transverse_shear_stiffness(thk, ang, mats, mdb)[0]
    G = r["G_msg"]
    print("%-9s %2d %8.4f | %11.4e %11.4e | %11.4e %11.4e | %9.2e" %
          (ln, len(thk), h, G[0, 0], G[1, 1], Gw[0, 0], Gw[1, 1], r["Ustar_rel"]))
""")

md(r"""
## Reading the results

* **`Ustar_rel` ≤ ~1e-2 everywhere** — the least-squares projection absorbs almost all of the
  second-order gradient energy into $\gamma^T G \gamma$; the RM wall law is close to
  asymptotically correct for these laminates.  Where it grows (thick soft-core walls), *no*
  choice of $G$ makes the RM form adequate — that is a model limit, not a fitting problem.
* **MSG $G$ vs Whitney** — for the foam-sandwich webs the MSG value is 2–3× softer.  The two
  answer different questions: Whitney is the complementary-energy shear flow of the laminate
  alone; the MSG $G$ is the value that makes the RM *plate model* reproduce the second-order
  asymptotic energy of the 3-D laminate.
* The `G_msg = None` gate: `rm_plate_msg` returns `None` when the fitted compliance
  $X=G^{-1}$ is not positive definite (degenerate placeholder materials can trigger this,
  e.g. the 10-Pa gelcoat of the bundled MH-104 YAML) — fall back to Whitney in that case.
* Theory: Yu 2003 Sec. 4 (Eqs. 55–60) and the equation crosswalk to the IJSS/CMAME twins in
  `docs/MITC_transverse_shear.md`; the 8×8 storage convention matches
  `examples/data/benchmark/st15_rm_plate_8x8.dat`.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}}
nbf.write(nb, OUT)
print("wrote", OUT, "(%d cells)" % len(C))
