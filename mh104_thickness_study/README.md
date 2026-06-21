# mh104 wall-thickness sweep — JAX-Kirchhoff shell vs FEniCS solid

Timoshenko 6×6 beam stiffness of the **mh104** composite airfoil cross-section over a wall-thickness
sweep, comparing the **JAX MSG-shell (Kirchhoff / C1-Hermite)** to the **VABS-validated FEniCS
2D-solid** reference. Both are referenced at the quarter-chord (mesh origin), order
`[EA, GA2, GA3, GJ, EI2, EI3]`.

## Cases
- **Thickness factor** `f ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}` — scales every lamina thickness
  (`materials.xml` lamina = base × f). Outer contour (OML) is fixed; the wall grows inward.
- **Shell reference surface**: `OML` (frac=0), `center` (0.5), `IML` (1.0).

## Method / key corrections found in this study
1. **CCW contour traversal** (`build_ref_yaml.py`, default). The shell must traverse the contour to
   match the PreVABS/XML baseline flow (`l34:l23` → top TE→LE = CCW), *not* the raw datapoint order
   (CW). A reversed traversal flips **e2** (mirror-images off-axis ply angles) **and e3 = e1×e2**
   (flips the layup stacking) → it was inflating GJ/EI2 by ~+20%. Fixed: GJ/EI2 → ~+3%.
   Verified element-by-element vs the solid: e2·e2 = e3·e3 = +1.000.
2. **k22 = 0 for flat 2-node elements** (`msg_mesh.mesh_curvature`). Curvature comes from mesh
   refinement (flat facets), not a spurious per-element adjacency estimate.
3. **IML offset fix** (`msg_mesh.offset_oml_to_iml`). The reference-surface offset now takes each
   junction node's thickness from the **skin-consensus direction**, not the average over the thick
   perpendicular web — otherwise thick-wall IML geometry folded (EI2 blew up to +228%).

## Results (diagonal % diff, shell − solid)
- **Thin walls (f=0.1):** EA +1.6, GA2 +1.9, GJ +2.2, EI2 +1.4, EI3 +3.4, **GA3 +17.3** (%). All ≤4%
  except GA3 — the strong Kirchhoff/Hermite result.
- **GA3 (transverse shear)** is the lone large residual (~+17–22%). A mesh-refinement study
  (`refine_and_test.py`, 131→1048 elems) shows it **plateaus** → it is the **Kirchhoff-shell model
  limit**, not discretization.
- **IML vs OML:** in the valid thin regime the IML is *better* than OML for EA/EI2/EI3 (less
  web–skin overlap double-counting), as expected.
- **Thickness:** agreement degrades with f (shell theory assumes thin walls): GA2 +7→+32%,
  GA3 +21→+67% by f=0.6. Thick-wall (f≥0.6) IML GA3/GJ remain the hardest terms.

## Files
- `results/C6_solid_f0NN.txt` — FEniCS solid 6×6 (reference), 6 files.
- `results/C6_shell_jax_{OML,center,IML}_f0NN.txt` — JAX-Kirchhoff shell 6×6, 18 files.
- `results/timo_shell_jax_summary.txt` — shell diagonals, all f/refs.
- `results/timo_comparison_tables.txt` — full shell-vs-solid tables (abs + %diff).
- `figures/oml_abs_{diagonal,coupling_1,coupling_2}.png` — all 21 Timo terms vs f, **shell OML vs solid**
  (absolute, enlarged).
- `figures/oml_pcterr_{diagonal,coupling_1,coupling_2}.png` — **% error** of all 21 terms vs solid (OML, enlarged).
- `figures/ref_orient_oml_center.png`, `iml_orient_debug.png` — reference-surface e1/e2/e3 (OML is the
  default; center/IML offset shown for debugging the thick-wall web-junction fold).
- `figures/orient_shell_f0NN.png` — e1/e2/e3 element orientation (shell).
- `figures/orient_cmp_f0NN_*.png` — solid-vs-shell orientation check (e2·e2/e3·e3 ≈ +1).

## Reproduce
```
# shell side (Windows JAX env C:\conda_envs\opensg_2_0_env):
python debug/build_ref_yaml.py connect f=0.20     # CCW mesh (default)
python debug/run_all_shell.py                      # all f: meshes, orient images, Timo @ 3 refs
python debug/rehomog.py                             # fast re-homogenize only
python debug/plot_and_tables.py ; python debug/plot_all_terms.py
# solid side (WSL FEniCS env opensg_env_v8):
python run_sweep.py            # f=0.2..1.0 solid + FEniCS-shell
python run_solid_extra.py      # fill missing solids (f=0.1/0.8/1.0)
# new factor solid (e.g. f=0.1): cases/make_f010.py (PreVABS) -> prevabs_mh104/convert_sg_to_yaml.py
```
