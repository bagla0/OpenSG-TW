# Station-15 dehomogenization benchmark data (BAR-URC)

Reference data for the RM cross-section dehomogenization tutorial
(`docs/tutorials/st15_dehomogenization.ipynb`).

| file | what |
|------|------|
| `bar_urc-15-t-0.in.SM` | VABS 3-D stress recovery at the solid Gauss points (the benchmark). Columns: `y2 y3 s11 s12 s13 s22 s23 s33` (VABS order); the loader reorders to `[S11,S22,S33,S23,S13,S12]`. |
| `solid.lp_sparcap_center_thickness_015.coords` | through-thickness path down the LP spar-cap CENTRE (valid dehom path). |
| `solid.circumferential_015.coords` | path around the section near the outer surface. |
| `solid.lp_sparcap_left_edge_thickness_015.coords` | cap/web-CORNER path (documented shell-reference-fold limit). |

The 1-D shell SG (`examples/data/1d_yaml/st15_shell.yaml`) and the VABS `.K`
(`examples/data/benchmark/st15_vabs.K`) complete the standalone tutorial inputs.
