IEA-22-280-RWT blade -- 1-D RM shell cross-section YAMLs
========================================================
One-dimensional (contour) Reissner-Mindlin shell structure-gene meshes for the
IEA-22 blade span stations used in the RM cross-section paper (Examples 3-5).

  shell_r020.yaml ... shell_r090.yaml   station r/R = 0.2 ... 0.9

Reference surface: outer mould line (OML).  Each file carries the closed airfoil
contour + shear webs as C0 quad-ring elements with the per-segment 8x8 wall law
(ABD + transverse shear), the ply layup by name, and the material database.

These are the 1-D inputs to the RM homogenizer (mitc_rm_segment/run_ring_indep.py)
and to the two-step dehomogenization (examples/TW-paper/xsec_paper/dehom_rm.py).
The matching 2-D solid VABS meshes (.sg) and VABS outputs (.K/.SM/.U) live in
examples/data/2d_yaml/IEA_VABS/.
