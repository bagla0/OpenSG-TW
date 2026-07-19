"""dbg_blade_equil.py -- WHICH pre-buckling N is right on the blade: the 3-D FE stress or the RM dehom?

Section equilibrium is the arbiter.  The applied beam moment at station i (FF[i], the exact resultant of the
SAME nodal traction the FE solves) must equal the internal moment carried by the axial membrane resultant:
    M_y(i) = -oint N11 * z ds          (z = flapwise coord, about the reduction point y=z=0)
Compute that integral from (a) the FE membrane N and (b) the dehom N, and compare both to FF[i][4].
Whichever integral matches FF is the physically correct pre-buckling stress.
Also checks global load equilibrium (total f vs FF[0])."""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

NSE = bi.NSE; MPER = bb.MPER; NSTA = bb.NSTA
bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
print("mesh %d nodes %d quads" % (len(nodes), len(quads)))
_fm = sb._flat_node_mask(nodes, quads)
print("drilling mask: COPLANAR_ONLY=%s -> flat %d / FOLD %d of %d nodes"
      % (sb._COPLANAR_ONLY, int(_fm.sum()), int((~_fm).sum()), len(_fm)))

f = bb.traction_load(nodes, quads)
fx = f.reshape(-1, 6)[:, :3]
print("=== global load ===")
print("  total applied force  (Fx,Fy,Fz) = %s" % np.array2string(fx.sum(0), precision=4))
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
print("  FF[0] (root beam force)         = %s" % np.array2string(FF[0], precision=4))
print("  -> FF[0][:3] must equal total applied force (same f).")

u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root)
Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)          # FE membrane N (local frame, N[0]=N11 spanwise)
Nr = bi.rm_N(bl, FF)                                        # RM dehom membrane N


def moment_from_N(Ne, p, i):
    """-oint N11 z ds over the section elements of span layer p, using station i geometry."""
    P = bl["Pk"][i]                                         # (Ntot,2) = (y_chord, z_flap)
    M = 0.0
    for se in range(NSE):
        a, b = int(bb.sec_elems[se, 0]), int(bb.sec_elems[se, 1])
        ds = np.linalg.norm(P[b] - P[a]); zmid = 0.5 * (P[a, 1] + P[b, 1])
        M += -Ne[p * NSE + se, 0] * zmid * ds
    return M


print("\n=== section equilibrium: internal moment vs applied FF ===")
print(" sta   FF_My(applied)     M_y(FE stress)    M_y(RM dehom)    FE/FF     RM/FF")
for i in [5, 15, 25, 35, 45]:
    p = min(i * MPER, bl["NS"] - 2)
    mfe = moment_from_N(Nf, p, i); mrm = moment_from_N(Nr, p, i); ff = FF[i][4]
    print("  %2d   %+.4e      %+.4e     %+.4e    %6.3f   %7.3f"
          % (i, ff, mfe, mrm, mfe / ff if ff else np.nan, mrm / ff if ff else np.nan))
