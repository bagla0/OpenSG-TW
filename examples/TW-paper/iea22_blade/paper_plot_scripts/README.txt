Paper plot scripts -- IEA r=0.2 dehomogenization (Example 5)
===========================================================
Archival copies of the Python scripts that generate the r=0.2 local-field
recovery figures in the RM cross-section paper.  The RUNNABLE originals live in
    examples/TW-paper/xsec_paper/
(they import sibling modules there and resolve data via relative paths); run them
from that directory with the opensg_2_0 environment.  These copies are kept here
so the IEA-blade example is self-documenting.

Supporting modules
  dehom_rm.py                RM-consistent two-step dehomogenization (C0 MITC-g23
                             ring; disp_at_points / stress_at_points / build_rm_bundle)
  oml_ring.py                OML/center 1-D ring loader + RM 6x6 (c6)

Figure generators (station r=0.2, load FF = BAR-URC critical shell case)
  emit_r020_dehom_paper.py   figures/r020_section_paths.png (section + 2 paths),
                             tab_rm/r020_homo.tex (RM vs VABS .K, all nonzero Cij),
                             figures/r020_disp.png (VABS .U warping)
  dehom_r020_figs.py         figures/dehom_r020_circumferential.png (circ stress line),
                             figures/dehom_r020_capleft.png (cap-through-thickness stress)
  dehom_r020_dispstress.py   figures/r020_stress_contour_cmp.png (in-plane stress
                             contour, VABS vs RM)
  r020_disp_contour.py       figures/r020_disp_contour_cmp.png (u1,u2,u3 contour,
                             VABS vs RM), r020_disp_circ.png, r020_disp_cap.png
  r020_iso_warp.py           figures/r020_iso_warp.png (isometric u1 on distorted mesh)

Individual line/path plots are also stored beside this folder in ../data/:
  r020_disp_circ.png, r020_disp_cap.png,
  dehom_r020_circumferential.png, dehom_r020_capleft.png
