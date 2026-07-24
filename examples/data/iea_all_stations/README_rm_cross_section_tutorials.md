# IEA-22 RM cross-section tutorials — bundled input data

This folder holds the **input data** for the two Reissner–Mindlin cross-section tutorials
that reproduce the *Composites Part B* paper on the IEA-22 MW reference blade
(mid-surface / center reference throughout):

- [`docs/tutorials/iea_r020_homo_dehom.ipynb`](../../../docs/tutorials/iea_r020_homo_dehom.ipynb) — `r/R=0.2` homogenization + 3-path dehomogenization
- [`docs/tutorials/iea_spanwise.ipynb`](../../../docs/tutorials/iea_spanwise.ipynb) — 51-station spanwise homogenization + recovery

Each tutorial is also a plain runnable script (`docs/tutorials/*.py`). Everything runs
**standalone from this repository** — no path points at any external machine.

## What the tutorials read (all under this folder)

| file(s) | role |
|---|---|
| `shell51/1d_yaml/iea_s00_shell.yaml … iea_s50_shell.yaml` | the 51 center-reference 1-D shell structure-genes (nodes + layup + materials, `reference: center`); `iea_s10` is the `r/R=0.2` station. The RM ring homogenization reads only these. |
| `dehom51/beamdyn/ff51_rmc_reform.dat` | per-station BeamDyn sectional forces/moments (VABS order); drives the two-step recovery. |
| `dehom51/beamdyn/iea51rmc_bd_driver.out` | BeamDyn nodal translations/rotations for the spanwise total-displacement transport. |
| `dehom51/out/VABS_iea51/iea_s10.sg.K` | VABS Timoshenko 6×6 at `r/R=0.2` (homogenization benchmark). |
| `dehom51/out/VABS_iea51/iea51vabs_bd_driver.out` | BeamDyn kinematics used to reconstruct VABS total local displacement at `r/R=0.2`. |
| `dehom51/out/dehom_vabs/iea_s10.circumferential.out`, `…lp_sparcap_left_thickness.out` | VABS local stress + displacement along the circumferential and spar-cap paths (path coordinates + benchmark). |
| `dehom51/out/cpb_r020_msgrm/data/junction_polyline_mid.dat` | the connected cap→junction→web polyline vertices. |
| `dehom51/benchmark/spanwise_vabs_landmarks.npz` | **pre-extracted** VABS landmark (stress, displacement, 6×6) for all 51 stations — ships in place of the multi-hundred-MB VABS field dumps. Built once by `dehom51/benchmark/_extract_landmarks.py` (kept for provenance; not needed at runtime). |

## Reproduce from scratch

```bash
pip install -e .[jax]          # numpy, scipy, jax[cpu], pypardiso, matplotlib, pyyaml
python docs/tutorials/iea_r020_homo_dehom.py
python docs/tutorials/iea_spanwise.py
```

The RM code itself lives in `opensg_jax/`, `mitc_rm_segment/`, and
`examples/TW-paper/xsec_paper/` (imported via the repo root); the tutorials add those to
`sys.path` automatically.
