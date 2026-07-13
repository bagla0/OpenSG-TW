---
name: prevabs-xsec
description: "Build a PreVABS 2-D composite cross-section (.xml -> .sg VABS-input mesh -> 2-D solid OpenSG YAML) from windIO/OpenFAST blade data, and diagnose/fix PreVABS meshing failures. Use when the user (who has PreVABS installed) wants a 2-D cross-section mesh/.sg/.xml at one or all stations, or is debugging a PreVABS gmsh error."
---

# PreVABS 2-D cross-section builder

Build 2-D cross-sections via the **PreVABS XML as the common pathway**, and fix meshing failures.
Companion of the `prevabs-xsec-builder` agent (transferable copy in `.claude/agents/`); full pipeline
detail in memory `ref_iea_all_stations`, `ref_prevabs_mh104_pipeline`, `ref_windio_converter`.

## Pipeline (XML is the hub)
`build_cross_section(blade,r)` → `emit_prevabs` → `<name>.xml` (+ materials.xml + `.dat`) →
`prevabs -i <name>.xml --vabs --hm` → `<name>.sg` (VABS input mesh) → `convert_sg_to_yaml` → 2-D solid
YAML. Also `emit_opensg_yaml` → 1-D shell YAML. **Ready driver:** `OpenSG_io/scripts/obtain_input_cs_files.py`
(and its copy in `examples/data/iea_all_stations/`) — `--windio --out --r --stations --types sg,1d,2d
(2d implies sg; 1d skips PreVABS) --mesh-size --jobs N --pv-timeout`. **`--out` defaults to the script's dir**
— run the data-folder copy or pass `--out`.

## `--vabs --hm` caveat
`--vabs` = VABS output FORMAT, `--hm` = required analysis-mode flag; VABS only runs with `-e/--execute`. So
`prevabs -i x.xml --vabs --hm` (no `-e`) just WRITES the `.sg` mesh. The user runs VABS on the `.sg` for `.K`.

## Failure modes → fixes
- **`gmsh mesh generation failed: Unable to recover the edge N on curve C (surface S)`** — a THIN
  feature/sliver at a near-degenerate section (airfoil-to-cylinder ROOT transition; sub-metre TIP). Fix:
  sweep **`--mesh-size`** (usually COARSER 0.02–0.04), or nudge/merge micro-close dividing points, or widen
  web attachment tolerance; at a truly degenerate station skip it and interpolate. Fast-failing (~seconds).
- Hangs → always use `--pv-timeout`. Binary not found → set `PREVABS_EXE`/`--prevabs`. Spurious thick-wall
  stiffening → 1-D shell contour must be MID-surface not OML (`ref_opensg_io_centerref`).

## XML knobs (to fix a run)
`general` (mesh_size, format), `baselines` (airfoil `.dat` contour), dividing points/segments (arc
breakpoints), `webs` (attachment arc + layup), `layups` (ply stacks), components (arc→layup). Biggest
lever = global `mesh_size`.

## windIO / OpenFAST
windIO v2 (preferred) → `load_blade`→`build_cross_section` gets contour+webs+layup (real airfoil stations,
no invented intermediates). OpenFAST/BeamDyn is the DOWNSTREAM beam model (6x6 + loads), not a section
geometry source — use it to check the homogenized 6x6 or feed a later GEBT run (`gebt-beam` skill).

## Deliver
`xml/ sg/ 2d_yaml/ 1d_yaml/` (separate folders) + a table (station, chord, webs, sg/2d/1d ok?, mesh_size,
failure notes). Never claim success without checking the `.sg`/YAML exists; report every failure.
