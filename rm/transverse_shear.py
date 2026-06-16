"""
Transverse-shear stiffness of a laminate (the G block of the RM 8x8 plate
stiffness) by the FSDT / 3D-equilibrium method -- the rigorous "shear-corrected"
value, NOT (5/6) sum(G t).

Physics (per direction; x = ply-1 / beam, y = ply-2 / tangent):
  pure bending    : sigma_11(z) = Q11(z) (z - z_na) kappa          (z_na = neutral axis)
  3D equilibrium  : d sigma_13/dz = - d sigma_11/dx
  -> shear flow   : sigma_13(z) = (g(z)/D_eff) V ,  g(z)=INT_bot^z Q11(zeta-z_na) dzeta
  energy equiv.   : F = D_eff^2 / INT g(z)^2 / G13(z) dz
The SAME g(z)/D_eff recovers the transverse-shear STRESS from the beam shear force.

Validates: isotropic plate -> F = (5/6) G h exactly.
"""
import numpy as np


def _ply_Q_and_G(E, G, nu, theta_deg):
    """In-plane Q11,Q22 (reduced) and transverse shear G13',G23' for a rotated ply."""
    E1, E2 = E[0], E[1]; G12, G13, G23 = G[0], G[1], G[2]
    v12 = nu[0]; v21 = v12 * E2 / E1
    den = 1 - v12 * v21
    Q11, Q22, Q12 = E1/den, E2/den, v12*E2/den
    Q66 = G12
    th = np.deg2rad(theta_deg); c, s = np.cos(th), np.sin(th)
    # in-plane Q_bar (need Qbar11, Qbar22 for the two bending directions)
    Qb11 = Q11*c**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*s**4
    Qb22 = Q11*s**4 + 2*(Q12+2*Q66)*s**2*c**2 + Q22*c**4
    # transverse shear (2x2) rotated: [G13', G23'] diag in ply axes -> rotate
    Gp = np.array([[G13, 0.0], [0.0, G23]])
    R2 = np.array([[c, s], [-s, c]])
    Gbar = R2 @ Gp @ R2.T                       # [[G13',G13-23],[.,G23']]
    return Qb11, Qb22, Gbar


def transverse_shear_stiffness(thick, angles, mat_names, mat_db, nz_per_ply=40,
                               coupled=False):
    """2x2 transverse-shear stiffness G_2x2 (rows/cols = [2g13, 2g23]) and the
    shear-flow shapes for stress recovery.  Returns (Gmat, recover) where
    recover(z, V) -> [sigma13, sigma23].

    ``coupled`` selects the transverse-shear constitutive route:
      * False (default) -- the FSDT / Whitney(1973) energy-equivalence stiffness,
        per direction with the diagonal G13'/G23', F = D_eff^2 / INT g^2/G dz.
      * True -- the coupling-aware ("no-shear-correction-factor") MSG form: the
        full 2x2 from the laminate complementary shear energy with the FULL
        through-thickness shear-flexibility Gbar^{-1}(z),
        F = M^{-1}, M_ab = INT ghat_a ghat_b [Gbar^{-1}]_ab dz, ghat = g/D_eff.
        Identical to the FSDT diagonal when Gbar(z) is diagonal (G13=G23, or any
        specially-orthotropic wall); it differs only when the rotated transverse
        shear couples (G13 != G23 with off-axis plies)."""
    nlay = len(thick)
    zb = np.concatenate([[0.0], np.cumsum(thick)])     # ply boundaries from OML=0
    htot = zb[-1]
    # fine z grid + per-point Qbar11,Qbar22,Gbar
    zs, Qb11, Qb22, Gb = [], [], [], []
    for k in range(nlay):
        zk = np.linspace(zb[k], zb[k+1], nz_per_ply, endpoint=False)
        q11, q22, gbar = _ply_Q_and_G(mat_db[mat_names[k]]['E'], mat_db[mat_names[k]]['G'],
                                      mat_db[mat_names[k]]['nu'], angles[k])
        for z in zk:
            zs.append(z); Qb11.append(q11); Qb22.append(q22); Gb.append(gbar)
    zs = np.array(zs); dz = htot/len(zs)
    Qb11 = np.array(Qb11); Qb22 = np.array(Qb22); Gb = np.array(Gb)  # (N,2,2)

    def direction(Q, Gss):
        z_na = np.sum(Q*zs*dz)/np.sum(Q*dz)            # neutral axis
        zc = zs - z_na
        D_eff = np.sum(Q*zc**2*dz)                     # bending stiffness
        g = np.cumsum(Q*zc*dz)                         # shear flow INT_bot^z Q (z-z_na)
        F = D_eff**2 / np.sum(g**2/Gss*dz)
        return F, g, D_eff

    Fx, gx, Dx = direction(Qb11, Gb[:, 0, 0])          # 2g13 : G13'
    Fy, gy, Dy = direction(Qb22, Gb[:, 1, 1])          # 2g23 : G23'
    if coupled:
        gh1, gh2 = gx/Dx, gy/Dy                        # normalized shear-flow shapes
        Ginv = np.linalg.inv(Gb)                       # (N,2,2) full shear flexibility
        M = np.array([[np.sum(gh1*gh1*Ginv[:, 0, 0]*dz), np.sum(gh1*gh2*Ginv[:, 0, 1]*dz)],
                      [np.sum(gh1*gh2*Ginv[:, 1, 0]*dz), np.sum(gh2*gh2*Ginv[:, 1, 1]*dz)]])
        Gmat = np.linalg.inv(M)                         # full 2x2 (couples if Gbar off-diag)
    else:
        Gmat = np.array([[Fx, 0.0], [0.0, Fy]])        # diagonal (orthotropic Y=0)

    def recover(z, V):                                 # transverse-shear stress at depth z
        i = min(int(z/htot*len(zs)), len(zs)-1)
        return np.array([gx[i]/Dx*V[0], gy[i]/Dy*V[1]])
    return Gmat, recover, (zs, gx/Dx, gy/Dy)


def plate_8x8(D6, Gmat):
    """assemble the RM 8x8 plate stiffness [[D, 0],[0, G]] (Y=0, orthotropic)."""
    P = np.zeros((8, 8)); P[:6, :6] = D6; P[6:, 6:] = Gmat
    return P


# ----------------------------------------------------------- validation
if __name__ == "__main__":
    E, nu, h = 3.44e9, 0.3, 0.2
    G = E/(2*(1+nu))
    mat = {"iso": {"E": [E, E, E], "G": [G, G, G], "nu": [nu, nu, nu]}}
    Gmat, rec, _ = transverse_shear_stiffness([h], [0.0], ["iso"], mat)
    print("Isotropic plate transverse-shear stiffness:")
    print(f"  F (FSDT)  = {Gmat[0,0]:.6e}")
    print(f"  (5/6) G h = {5/6*G*h:.6e}")
    print(f"  ratio     = {Gmat[0,0]/(5/6*G*h):.6f}   (should be 1.000)")
    # 3-ply glass/carbon/glass-ish sanity
    mat2 = {"a": {"E": [4e10, 1e10, 1e10], "G": [4e9, 4e9, 3.5e9], "nu": [0.3, 0.3, 0.4]}}
    Gm2, _, _ = transverse_shear_stiffness([0.004, 0.07, 0.002]*1, [0, 0, 0]*1,
                                           ["a", "a", "a"], mat2)
    print("\n3-ply UD-ish: G =", np.array2string(Gm2, precision=3))
