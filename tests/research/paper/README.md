# Journal paper — thin-walled MSG shell vs solid (anisotropic tube)

`tw_msg_tube.tex` — Elsevier `elsarticle` paper, targeted at *Thin-Walled Structures*.

## Build
No local TeX is installed on this machine. Easiest route:

- **Overleaf**: New Project → Upload Project → upload this `paper/` folder
  (the `figures/` subfolder is included). Overleaf ships `elsarticle.cls`; it
  compiles as-is with pdfLaTeX. No bibtex needed (the bibliography is inline
  `thebibliography`).
- **Local TeX Live / MiKTeX**: `pdflatex tw_msg_tube` (run twice for refs).

## Figures (`figures/`)
| file | content |
|---|---|
| `fig_mesh.png` | shell 1D mesh + material frame; solid annulus quad mesh |
| `fig_orientation.png` | e2/e3 material frame around the ring |
| `fig_dehom.png` | recovered 3D stress, shell vs solid, top path |
| `fig_strain.png` | strain-driven σ11(z), shell vs solid |
| `fig_convergence.png` | thickness convergence (h/R → 0) |
| `fig_oop.png` | out-of-plane stress, Kirchhoff vs RM vs solid |

## Before submission — fill in / verify
- **Author list, ORCID, affiliation, co-authors/advisor** (currently a single
  placeholder author with a `% TODO`).
- Confirm the **published benchmark citation** for the anisotropic tube values
  in Table 2 / Section 3 and add it to the reference list.
- Add a **CRediT** author-contributions statement and funding/acknowledgements.
- The DOIs/volume numbers in `thebibliography` are representative; verify each.
