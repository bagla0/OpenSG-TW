# MITC-RM tapered-segment (3D-SG) pipeline — prismatic cylinder

Reissner–Mindlin (MITC) Timoshenko-6×6 homogenization of a **surface-quad shell
segment**, solved with the OpenSG **3D-SG tapered-segment** method: the two end
cross-sections are extracted as separate 1-D boundary SGs (via dolfinx), solved
on their own, and their warping (V0, V1) is transferred as Dirichlet constraints
onto the segment before the segment is solved.

Reference: IMECE2025 paper + `opensg/mesh/segment.py`, `opensg/core/shell.py`.
The RM/MITC formulation itself comes from the JAX side (`opensg_jax/fe_jax/
msg_rm_timo.py`, `strip_RM.py`) — the FEniCS reference is Kirchhoff/solid only.

## Hybrid architecture (why 3 stages / 2 environments)

| Stage | File | Env | Role |
|------|------|-----|------|
| 1 | `make_cylinder_segment.py` | Windows `opensg_2_0_env` | build the surface-quad segment YAML |
| 2 | `extract_boundaries_dolfinx.py` | **WSL `opensg_env_v8`** (dolfinx 0.8.0) | dolfinx `create_submesh` → left/right 1-D rings + material frame + node maps → `.npz` |
| 3 | `solve_segment_jax.py` | Windows `opensg_2_0_env` | RM/MITC boundary solve → V0/V1 → segment solve → Timoshenko 6×6 |

dolfinx is used only where it is genuinely needed — generically slicing the two
end cross-sections off an (in general unstructured, tapered) surface mesh and
carrying the per-element frame/material onto them. The RM/MITC solve reuses the
validated JAX code.

## Conventions (kept identical across all stages)

- **Beam axis = x**; cross-section in **(y, z)**; nodes/elements **1-indexed** in YAML.
- Per-element frame in `elementOrientations` = `[e1(3), e2(3), e3(3)]` with
  `e1` = axial, `e2` = hoop tangent, `e3` = **inward** normal.
- Reference surface = wall **mid-surface** (center reference): the symmetric iso
  ply has B ≈ 0; the `[45/-45]` stack keeps a genuine B16/B26 coupling.

## Test cases (the single-cell tubes validated earlier as 1-D cross-sections)

- **isotropic** `[0]`, E = 70 GPa, ν = 0.3      (cf. `cylinder_study/`)
- **anisotropic** `[45/-45]`, E1 = 37, E2 = E3 = 9 GPa, G = 4 GPa, ν = 0.3  (cf. `aniso_tube/`)

R = 1.0, NC = 160 hoop quads, NL = 3 axial quads, L = 1.5; wall t = h/R · R for
h/R ∈ {0.05, 0.1, 0.2}.

## Run

```bash
# stage 1 (Windows)
python make_cylinder_segment.py            # -> meshes/seg_{iso,aniso}_hR*.yaml

# stage 2 (WSL)
conda activate opensg_env_v8
python extract_boundaries_dolfinx.py meshes/seg_iso_hR0.1.yaml out/seg_iso_hR0.1.npz

# stage 3 (Windows) — TODO
python solve_segment_jax.py out/seg_iso_hR0.1.npz
```

## Stage-3 design (JAX MITC-RM solver — in progress)

1. **Material.** Per-layup ABD 6×6 (through-thickness 1-D SG, `msg_materials.
   compute_ABD_matrix`) + transverse-shear 2×2 for the RM `G`-energy.
2. **Boundary SGs.** Each end ring is a 1-D RM/MITC cross-section: reuse
   `msg_rm_timo.assemble_all` + `build_C_Psi` + the element-agnostic
   `msg_solver` KKT/V1 pipeline to get `V0` (4 EB modes) and `V1s` (Timoshenko).
   For the prismatic cylinder left ≡ right.
3. **Segment element.** A 2-D **MITC Reissner–Mindlin shell** element on the
   curved quad surface (5 DOF/node `[w1,w2,w3,ω1,ω2]`; bilinear N(ξ,η); 2×2
   Jacobian; MITC4 mid-edge tying of the transverse shears to cure locking;
   ABD material). This is the one genuinely new element vs. the existing code.
4. **Transfer.** Scatter the ring `V0/V1s` onto the segment boundary DOFs
   (`L_node2seg`/`R_node2seg`) as Dirichlet BCs — replaces periodicity and fixes
   the rigid-body modes.
5. **Solve + recover.** Segment `V0` then `V1s`; assemble the Timoshenko 6×6 via
   `B_tim / C_tim / Q_tim / Ginv / Y_tim` (identical algebra to solid & shell).

## Validation

Prismatic ⇒ both ends identical and the taper (`Γ_l`) terms vanish, so the
segment **Timoshenko 6×6 must equal the single 1-D cross-section 6×6** (iso and
aniso). That identity is the self-check before moving to real tapered blade
segments.

## Status

- [x] branch `mitc-rm-surface-quad`
- [x] stage 1 — mesh generator + 6 meshes
- [x] stage 2 — dolfinx boundary extraction (verified bundle: rings at x=0 / x=1.5, R=1.0, frames OK)
- [ ] stage 3 — JAX MITC-RM boundary + segment solve
- [ ] validation (segment 6×6 == cross-section 6×6)
