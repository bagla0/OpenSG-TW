# Tapered-segment homogenization — shell vs solid, whole blade

A reproducible pipeline that, for any span segment `[r1, r2]` of a windIO blade, builds
the **3-D solid** and the **equivalent 3-D shell** SG (plus their end cross-sections) and
computes the **Timoshenko 6×6** of each with the two homogenizers, then compares them.

Everything runs in **one environment** — the server `opensg_2_0` conda env, which carries
`jax` (shell), `dolfinx`/`opensg` (FEniCS solid, merged in), and `windIO` (mesh source):

```
~/miniconda3/envs/opensg_2_0/bin/python examples/taper/<script>.py ...
```

The blade windIO lives at `examples/data/windio/IEA-22-280-RWT.yaml`; the mesh generator
is `third_party/OpenSG_io` (opensg_io). Valid airfoil range: **r = 0.05 … 0.95** (the root
cylinder r=0 and the tip r=1 have no webbed airfoil section).

## Scripts (run in order; each is standalone)

| # | script | does |
|---|--------|------|
| 1 | `1_generate_solid_mesh.py <r1> <r2> <out>` | opensg_io → **solid** hex segment + 2-D solid boundary sections |
| 2 | `2_generate_shell_mesh.py <r1> <r2> <out>` | opensg_io → **shell** quad segment (OML ref) + 1-D shell boundary contours |
| 3 | `3_get_beam_props_from_solid_boundary.py <yaml>` | FEniCS 2-D solid cross-section 6×6 |
| 4 | `4_get_beam_props_from_solid_segment.py <yaml>` | FEniCS 3-D solid segment 6×6 (+ end rings) |
| 5 | `5_get_beam_props_from_shell_boundary.py <yaml>` | JAX MITC-RM 1-D ring 6×6 |
| 6 | `6_get_beam_props_from_shell_segment.py <yaml>` | JAX MITC-RM 3-D shell segment 6×6 (+ end rings) |

Every Timo script prints **DOF used, wall time, and the Timoshenko 6×6**, and saves a
`*_timo.npz`.

## Whole-blade sweep

```
python examples/taper/run_blade_sweep.py            # segments 0.05→0.95 (0.1 wide)
python examples/taper/run_blade_sweep.py 0.2 0.3    # one segment
```

For each segment it runs 1–6, writes a per-segment `comparison.dat` (taper + both boundary
rings, solid + shell + %-error 6×6), and stages a **separate folder per segment** into the
OneDrive deliverable (`SWEEP_OUT` env, default `~/OneDrive_.../IEA_blade_taper_sweep/`),
along with `SUMMARY_diag_error_vs_span.dat` tracking the diagonal error root→tip.

## Mesh conventions (opensg_io)
- **Ply-conforming** through-thickness hex layers (sandwich skins meshed exactly).
- **NuMAD/VABS orientation**: e1 = beam axis root→tip, e3 = inward normal, e2 = e3×e1.
- **Shell reference = OML** (coincident with the solid outer ring); pair with `frac=0` ABD.
- Span-interpolated per-bay layup with ply drops.
