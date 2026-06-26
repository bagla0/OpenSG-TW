"""OpenFAST / BeamDyn <-> OpenSG bridge.

BeamDyn distributes a 6x6 sectional STIFFNESS (and MASS) matrix per blade station. Its DOF order is
  [shear_x, shear_y, axial(EA), bend_x(EI), bend_y(EI), torsion(GJ)]
whereas the OpenSG/VABS Timoshenko 6x6 used here is
  [EA(axial), GA2(shear), GA3(shear), GJ(torsion), EI2(bend), EI3(bend)].

read_beamdyn_blade()  -> etas + 6x6 stiffness/mass per station (BeamDyn order), to use as a REFERENCE.
beamdyn_to_timo()     -> reorder a BeamDyn 6x6 into the OpenSG Timoshenko order (axial/torsion are exact;
                         the shear pair and the bending pair are matched to OpenSG 2/3 & 5/6 by magnitude:
                         the larger bending = edgewise = EI3, the larger shear = GA2).
timo_to_beamdyn()     -> inverse, to EXPORT an OpenSG-homogenized 6x6 back to BeamDyn order.
write_beamdyn_blade() -> write a BeamDyn blade input from homogenized stiffness (+ optional mass), so the
                         OpenSG cross-section results can drive an OpenFAST aeroelastic run.
"""
import numpy as np

# OpenSG-Timo index <- BeamDyn index, for the unambiguous DOFs (axial, torsion)
_BD_AXIAL, _BD_TORSION = 2, 5


def _floats(line):
    out = []
    for tok in line.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def read_elastodyn_blade(path):
    """Parse an OpenFAST ElastoDyn blade file (the distributed BLADE DATA). Returns a dict of arrays:
    BlFract (span fraction), PitchAxis, StrcTwst (deg), BMassDen (kg/m), FlpStff (EI flapwise, N m^2),
    EdgStff (EI edgewise, N m^2). These map to OpenSG EI2 (flap) and EI3 (edge) for validation."""
    lines = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if "DISTRIBUTED BLADE PROPERTIES" in ln) + 1
    rows, started = [], False
    for ln in lines[i:]:
        f = _floats(ln)
        if len(f) >= 6:
            rows.append(f[:6]); started = True
        elif started:
            break
    a = np.array(rows)
    return dict(BlFract=a[:, 0], PitchAxis=a[:, 1], StrcTwst=a[:, 2], BMassDen=a[:, 3],
                FlpStff=a[:, 4], EdgStff=a[:, 5])


def elastodyn_at(ed, r):
    """Interpolate (EI_flap, EI_edge, mass_per_len, twist_deg) from an ElastoDyn blade dict at span r."""
    g = ed["BlFract"]
    return (float(np.interp(r, g, ed["FlpStff"])), float(np.interp(r, g, ed["EdgStff"])),
            float(np.interp(r, g, ed["BMassDen"])), float(np.interp(r, g, ed["StrcTwst"])))


def read_beamdyn_blade(path):
    """Parse a BeamDyn blade input. Returns (etas, K_list, M_list) with each K/M a 6x6 ndarray in
    BeamDyn DOF order. Stations are the eta (non-dimensional span) values."""
    lines = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if "DISTRIBUTED PROPERTIES" in ln) + 1
    etas, Ks, Ms = [], [], []
    n = len(lines)
    while i < n:
        f = _floats(lines[i])
        if len(f) == 1:                                    # station header = single eta value
            eta = f[0]; i += 1
            K = np.array([_floats(lines[i + r]) for r in range(6)]); i += 6
            while i < n and not _floats(lines[i]):
                i += 1
            M = np.array([_floats(lines[i + r]) for r in range(6)]); i += 6
            if K.shape == (6, 6) and M.shape == (6, 6):
                etas.append(eta); Ks.append(K); Ms.append(M)
        else:
            i += 1
    return np.array(etas), Ks, Ms


def _perm_bd_to_timo(Kbd):
    """OpenSG-Timo index -> BeamDyn index. Axial/torsion exact; shear & bending pairs matched by magnitude
    (larger bending = edgewise = EI3; larger shear = GA2)."""
    sx, sy = (0, 1) if Kbd[0, 0] >= Kbd[1, 1] else (1, 0)   # GA2 (larger) <- sx, GA3 <- sy
    # bending: EI2 (smaller, flapwise) and EI3 (larger, edgewise)
    bsmall, blarge = (3, 4) if Kbd[3, 3] <= Kbd[4, 4] else (4, 3)
    return [_BD_AXIAL, sx, sy, _BD_TORSION, bsmall, blarge]   # Timo[i] <- BeamDyn[perm[i]]


def beamdyn_to_timo(Kbd):
    """Reorder a BeamDyn 6x6 into OpenSG Timoshenko order [EA, GA2, GA3, GJ, EI2, EI3]."""
    Kbd = np.asarray(Kbd, float)
    p = _perm_bd_to_timo(Kbd)
    return Kbd[np.ix_(p, p)]


def timo_to_beamdyn(Ktimo):
    """Reorder an OpenSG Timoshenko 6x6 [EA, GA2, GA3, GJ, EI2, EI3] into BeamDyn order
    [shear_x, shear_y, axial, bend_x, bend_y, torsion]."""
    Ktimo = np.asarray(Ktimo, float)
    # Timo index for each BeamDyn slot: sx<-GA2(1), sy<-GA3(2), axial<-EA(0), bend_x<-EI2(4), bend_y<-EI3(5), tors<-GJ(3)
    q = [1, 2, 0, 4, 5, 3]
    return Ktimo[np.ix_(q, q)]


def write_beamdyn_blade(etas, K_timo_list, M_list=None, out_path="blade_BeamDyn_Blade.dat",
                        damp=(0.003, 0.002, 0.002, 0.002, 0.003, 0.002)):
    """Write a BeamDyn blade input from OpenSG-homogenized Timoshenko stiffness (one 6x6 per eta). M_list is
    optional 6x6 mass matrices (BeamDyn order); zeros if omitted. Exports OpenSG results to OpenFAST."""
    etas = list(etas)
    n = len(etas)

    def block(M):
        return "\n".join("\t" + "\t".join("% .16e" % M[r, c] for c in range(6)) for r in range(6))
    out = [" ------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE (OpenSG-homogenised) -------",
           " Generated by opensg_io / openfast_io.py",
           " ---------------------- BLADE PARAMETERS --------------------------------------",
           "%d   station_total    - Number of blade input stations (-)" % n,
           " 1   damp_type        - Damping type: 0: no damping; 1: damped",
           "  ---------------------- DAMPING COEFFICIENT------------------------------------",
           "   mu1        mu2        mu3        mu4        mu5        mu6",
           "   (-)        (-)        (-)        (-)        (-)        (-)",
           "  " + " ".join("%g" % d for d in damp),
           " ---------------------- DISTRIBUTED PROPERTIES---------------------------------"]
    for k in range(n):
        Kbd = timo_to_beamdyn(K_timo_list[k])
        Mbd = np.asarray(M_list[k], float) if M_list is not None else np.zeros((6, 6))
        out += ["\t %f " % etas[k], block(Kbd), "", block(Mbd), ""]
    open(out_path, "w").write("\n".join(out) + "\n")
    return out_path


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    etas, Ks, Ms = read_beamdyn_blade(path)
    lbl = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
    print("BeamDyn blade '%s': %d stations" % (path, len(etas)))
    print("  eta    " + "".join("%12s" % L for L in lbl) + "   (OpenSG-Timo order)")
    for eta, K in zip(etas, Ks):
        T = beamdyn_to_timo(K)
        print("  %.3f  " % eta + "".join("%12.4e" % T[i, i] for i in range(6)))
