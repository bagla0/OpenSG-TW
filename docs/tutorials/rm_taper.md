# Tapered thin-walled beams (RM_taper)

This tutorial reproduces the results of the paper *"Timoshenko Beam Modeling of Tapered
Thin-Walled Composite Structures Using the Reissner–Mindlin Model"* from the standalone
scripts in [`examples/RM_taper/`](https://github.com/bagla0/OpenSG-TW/tree/main/examples/RM_taper).
Each script loads a 1-D shell mesh YAML and prints the equivalent-beam Timoshenko `6×6`
(`C^b`) for the **boundary ring** and the **tapered segment**, compared term by term against a
conforming 3-D FEniCS solid reference — covering the paper's **12 cases** (3 geometries × 2
wall thicknesses × {boundary, taper}), single `[-45°]` ply.

```bash
python examples/RM_taper/circle.py     # circular tube
python examples/RM_taper/square.py     # square tube
python examples/RM_taper/ellipse.py    # webbed ellipse (multi-cell, 3 shear webs)
```

## The RM shell

A single **6-DOF independent-drilling** Reissner–Mindlin shell element homogenizes both the
boundary rings and the tapered interior. The transverse-shear scheme is set by
`_rm_common.shear_for(stage, tR)`:

| stage | thin `t/R ≤ 0.02` | thick `t/R > 0.02` |
|-------|-------------------|--------------------|
| tapered segment | full 2×2 Gauss | full 2×2 Gauss |
| boundary ring   | γ₂₃-tie (`mitc4_g23`) | full |

The independent-ω₃ element is **locking-free under full integration**, so the segment uses
full integration at every thickness; assumed-strain (MITC) tying is *not* used there because
it aliases the drilling-carried shear on flat walls (the square thin taper collapses to
`GA ≈ −47%` under MITC vs `−1.7%` under full integration). For the boundary ring, circle and
square are indifferent to the scheme; the webbed multi-cell ring uses a γ₂₃-tie on thin walls
(`GA₂ ≈ −17%` vs full's `+29%`).

## Results — RM shell vs. conforming solid (diagonal `%`-error)

| case | EA | GA₂ | GA₃ | GJ | EI₂ | EI₃ |
|------|---:|----:|----:|---:|----:|----:|
| circle thin — boundary | +2.2 | +3.0 | +3.0 | +3.0 | +2.2 | +2.2 |
| circle thin — taper | +0.8 | +5.1 | +5.1 | +0.0 | +2.0 | +2.0 |
| circle thick — boundary | +2.2 | +2.2 | +2.2 | +2.4 | +1.5 | +1.5 |
| circle thick — taper | +0.3 | +3.5 | +3.5 | −1.1 | +0.9 | +0.9 |
| square thin — boundary | +0.8 | +1.5 | +1.5 | +1.4 | +0.5 | +0.5 |
| square thin — taper | +1.3 | −1.7 | −1.7 | −4.4 | +2.3 | +2.3 |
| square thick — boundary | +0.8 | −1.8 | −1.8 | −0.2 | −0.3 | −0.3 |
| square thick — taper | +0.8 | +1.9 | +1.9 | −6.1 | +1.7 | +1.7 |
| ellipse thin — boundary | +3.4 | −16.8 | +3.3 | +3.0 | +1.9 | +4.9 |
| ellipse thin — taper | +2.5 | −2.5 | +28.7 | +0.7 | +0.5 | +2.9 |
| ellipse thick — boundary | +13.2 | −3.9 | +5.6 | +3.7 | +12.6 | +10.6 |
| ellipse thick — taper | +13.4 | −4.5 | +13.9 | +1.4 | +12.6 | +7.9 |

**Circle and square** reproduce every Timoshenko stiffness within a few percent — the square
thin taper `GA = −1.7%` is the paper's headline flat-wall result, versus the `−40%` of the
classical eliminated-drilling element. The **webbed ellipse** is the demanding multi-cell case:
its diagonal is recovered well except the thin-taper `GA₃` (`+28.7%`) and the thin-boundary
`GA₂` (`−17%`), while the thick walls over-predict `EA`/`EI` by `~13%` — the T-junction
material double-count, where each web mid-line carries a thickness that overlaps the skin's.

## Data

| geometry | shell mesh | solid ref (`L` = boundary, `seg` = taper) |
|----------|------------|--------------------------------------------|
| circle | `examples/data/taper_study/meshes/shell_<regime>_m45_aR070.yaml` | `examples/data/benchmark/taper_study_solid_m45.npz` |
| square | `examples/data/taper_square/meshes/shell_<regime>_m45_aR070.yaml` | `examples/data/benchmark/taper_square_solid_m45.npz` |
| ellipse | `examples/data/rm_taper_ellipse/meshes/shell_<regime>_m45.yaml` | `examples/data/benchmark/ellipse_solid_m45.npz` |
