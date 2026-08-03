'''gen_glb_from_beamdyn.py  --  write VABS .glb (global-response / recovery input) files, ONE PER
CROSS-SECTION, from a BeamDyn .out.  VABS then does dehomogenization: `vabs <name>.sg 2 0` reads the
sibling <name>.sg.glb to recover the 3-D stress/strain fields for that station.

------------------------------------------------------------------------------------------------
.glb FORMAT (decoded from the BAR-URC reference .glb; VABS Timoshenko recovery input):
  line 1     :  u1 u2 u3                    beam reference-point displacement (VABS frame)   [m]
  lines 2-4  :  3x3 direction-cosine matrix C_Bb  (rotation of the deformed section triad)
  line 5     :  F1 M1 M2 M3                 CLASSICAL resultants: axial, torsion, 2x bending  [N, N-m]
  line 6     :  F2 F3                       TRANSVERSE SHEAR resultants (Timoshenko)          [N]
  lines 7-10 :  0 0 0 0 0 0   (x4)          sectional-load derivatives (0 for static recovery)

------------------------------------------------------------------------------------------------
BeamDyn -> VABS axis remap (beam-axis swap; same map as OpenSG stress_recov.py:155 and the 6x6
transform B=[[0,0,1],[0,-1,0],[1,0,0]]).  Displacement/rotation use the GLOBAL ('r') BeamDyn frame;
the sectional force/moment resultants use the section-LOCAL ('L') frame (VABS recovery is in the
section frame, and the DCM rotates it back to global):
  disp    (u1,u2,u3)_VABS = ( TDzr, -TDyr,  TDxr)      global 'r' frame
  rot     (c1,c2,c3)_VABS = ( RDzr, -RDyr,  RDxr)      Wiener-Milenkovic parameters, 'r' frame
  force   (F1,F2,F3)_VABS = ( FzL,  -FyL,   FxL)       section-local 'L' frame
  moment  (M1,M2,M3)_VABS = ( MzL,  -MyL,   MxL)       section-local 'L' frame

------------------------------------------------------------------------------------------------
3x3 ROTATION MATRIX  --  Wang et al., "Geometric Exact Beam Theory with Wiener-Milenkovic
Parameters", SDM 2013, Section "Wiener-Milenkovic Parameters" (p.3, Rodrigues rotation tensor).
BeamDyn's RD* outputs are the Wiener-Milenkovic (WM) vectorial rotation parameters c=(c1,c2,c3),
NOT small angles, so the direction-cosine matrix is the EXACT tensor (Wang Eq. 12):

    c0 = 2 - (c1^2 + c2^2 + c3^2)/8
    C  = 1/(4 - c0)^2 *
         [ c0^2 + c1^2 - c2^2 - c3^2 ,  2(c1 c2 - c0 c3)         ,  2(c1 c3 + c0 c2)          ;
           2(c1 c2 + c0 c3)          ,  c0^2 - c1^2 + c2^2 - c3^2 ,  2(c2 c3 - c0 c1)          ;
           2(c1 c3 - c0 c2)          ,  2(c2 c3 + c0 c1)          ,  c0^2 - c1^2 - c2^2 + c3^2 ]

This reduces to the linearized skew form  [[1,-c3,c2],[c3,1,-c1],[-c2,c1,1]]  of
stress_recov.py:164 when |c| << 1 (hence the .glb diagonal ~0.99999889, not exactly 1).
VABS writes C_Bb; per the GEBT/VABS convention C_Bb = (C_bB)^T, so we output C (transpose if needed
to match the reference .glb -- the --validate-barurc mode checks the exact orientation).

    python gen_glb_from_beamdyn.py --bd iea51_solid_bd_driver.out --sgdir ../shell51/sg_v201 --glbdir out_glb
    python gen_glb_from_beamdyn.py --validate-barurc        # reproduce the BAR-URC .glb and diff
'''
import argparse
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------- Wiener-Milenkovic -> DCM (Wang SDM 2013)
def wm_to_dcm(c):
    c1, c2, c3 = c
    c0 = 2.0 - (c1 * c1 + c2 * c2 + c3 * c3) / 8.0
    d = (4.0 - c0) ** 2
    C = np.array([
        [c0 * c0 + c1 * c1 - c2 * c2 - c3 * c3, 2 * (c1 * c2 - c0 * c3), 2 * (c1 * c3 + c0 * c2)],
        [2 * (c1 * c2 + c0 * c3), c0 * c0 - c1 * c1 + c2 * c2 - c3 * c3, 2 * (c2 * c3 - c0 * c1)],
        [2 * (c1 * c3 - c0 * c2), 2 * (c2 * c3 + c0 * c1), c0 * c0 - c1 * c1 - c2 * c2 + c3 * c3],
    ]) / d
    return C


# ---------------------------------------------------------- read a BeamDyn .out (converged row)
def read_beamdyn(path):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(lines):
        if l.strip().startswith('Time'):
            hdr = l.split()
            data = np.array([r.split() for r in lines[i + 2:]], float)
            row = data[-1]                                    # converged static solution
            return {h: row[j] for j, h in enumerate(hdr)}, hdr
    raise ValueError('no header in %s' % path)


def n_nodes(hdr):
    return sum(1 for h in hdr if h.endswith('_TDxr') and h.startswith('N'))


# BeamDyn -> VABS axis map (beam-axis swap). Vectors map as B @ v = (v3, -v2, v1);
# a finite ROTATION maps as the SIMILARITY transform  C_v = B C_bd B^T  (NOT a component swap of
# the Wiener-Milenkovic vector, which only agrees for small angles).
B = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float)


def station_state(d, k):
    '''Per BeamDyn output node k (1-based): RAW BeamDyn-frame disp, WM rot params, force, moment.'''
    p = 'N%03d_' % k

    def g(name, alt=None):
        if p + name in d:
            return d[p + name]
        return d[p + alt] if alt else 0.0
    TD = np.array([g('TDxr'), g('TDyr'), g('TDzr')])         # displacement  (global IEC 'r' frame)
    RD = np.array([g('RDxr'), g('RDyr'), g('RDzr')])         # Wiener-Milenkovic rotation ('r' frame)
    # Sectional stress resultants for VABS recovery must be in the section-LOCAL (convecting) frame,
    # i.e. BeamDyn's 'L' outputs (FxL.. / MxL..), NOT the global 'r' resultants -- the .glb line 2-4
    # DCM then rotates the section-local recovered field into the global frame.  'r' is only a fallback.
    FF = np.array([g('FxL', 'Fxr'), g('FyL', 'Fyr'), g('FzL', 'Fzr')])
    MM = np.array([g('MxL', 'Mxr'), g('MyL', 'Myr'), g('MzL', 'Mzr')])
    return TD, RD, FF, MM


def to_vabs(TD, RD, FF, MM, transpose=True):
    '''Map raw BeamDyn-frame state to the VABS frame. Rotation via similarity C_v = B C_bd B^T;
    VABS stores C_bB = (C_Bb)^T so transpose=True by default (validated vs the BAR-URC .glb).'''
    u = B @ TD
    F = B @ FF
    M = B @ MM
    C_bd = wm_to_dcm(RD)
    C = B @ C_bd @ B.T
    if transpose:
        C = C.T
    return u, C, F, M


def write_glb(path, u, C, F, M):
    with open(path, 'w') as f:
        f.write('%.10g %.10g %.10g\n' % (u[0], u[1], u[2]))
        for r in range(3):
            f.write('%.16g %.16g %.16g\n' % (C[r, 0], C[r, 1], C[r, 2]))
        f.write('%.10g %.10g %.10g %.10g\n' % (F[0], M[0], M[1], M[2]))   # F1 M1 M2 M3
        f.write('%.10g %.10g\n' % (F[1], F[2]))                          # F2 F3
        for _ in range(4):
            f.write('0 0 0 0 0 0\n')


# ---------------------------------------------------------- main + BAR-URC validation
def gen_all(bd_out, sgdir, glbdir, stem_fmt):
    d, hdr = read_beamdyn(bd_out)
    N = n_nodes(hdr)
    os.makedirs(glbdir, exist_ok=True)
    print('BeamDyn %s : %d output nodes' % (os.path.basename(bd_out), N))
    for k in range(1, N + 1):
        TD, RD, FF, MM = station_state(d, k)
        u, C, F, M = to_vabs(TD, RD, FF, MM)
        out = os.path.join(glbdir, stem_fmt % (k - 1))       # node k (1-based) -> station k-1
        write_glb(out, u, C, F, M)
    print('wrote %d .glb files -> %s' % (N, glbdir))


def validate_barurc():
    '''Reproduce the BAR-URC .glb from bd_bar_urc.out and diff against the stored reference,
    to lock down the exact 3x3 rotation + force layout.'''
    cand_bd = ['/home/roger/a/bagla0/GIT_CLONE_opensg_20251021/OpenSG/examples/data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_.out',
               '/home/roger/a/bagla0/OpenSG/examples/data/Solid_3DSG/bar_urc_npl_1_ar_5-segment_.out',
               '/home/roger/a/bagla0/home/solid_segmentzip/solid_segment/bd_bar_urc.out',
               '/home/roger/a/bagla0/OpenSG/examples/data/3D_bd_bar_urc.out',
               '/home/roger/a/bagla0/OpenSG/examples/data/bd_bar_urc.out']
    bd = next((p for p in cand_bd if os.path.exists(p)), None)
    ref_dir = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/2d_yaml/bar_urc_glb'
    if not bd or not os.path.isdir(ref_dir):
        print('validation inputs not found (bd=%s, ref=%s)' % (bd, ref_dir)); return
    d, hdr = read_beamdyn(bd)
    N = n_nodes(hdr)
    print('bd_bar_urc.out: %d nodes; comparing my .glb vs bar_urc-<k>-t-0.in.glb' % N)
    for k in range(1, N + 1):
        refp = os.path.join(ref_dir, 'bar_urc-%d-t-0.in.glb' % (k - 1))
        if not os.path.exists(refp):
            continue
        TD, RD, FF, MM = station_state(d, k)
        u, C, F, M = to_vabs(TD, RD, FF, MM, transpose=False)     # C = B C_bd B^T
        ref = [ln.split() for ln in open(refp).read().splitlines() if ln.strip()]
        Cref = np.array([[float(x) for x in ref[r]] for r in (1, 2, 3)])
        dC = np.max(np.abs(C - Cref))
        dCt = np.max(np.abs(C.T - Cref))
        uref = np.array([float(x) for x in ref[0]])
        du = np.max(np.abs(u - uref))
        tag = 'C=' if dC < 1e-6 else ('C^T=' if dCt < 1e-6 else 'MISMATCH')
        print('  node %2d: |dC|=%.2e  |dC^T|=%.2e  |du|=%.2e (u=%.3g vs ref %.3g)  %s'
              % (k, dC, dCt, du, u[2], uref[2], tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bd', help='BeamDyn .out (e.g. iea51_solid_bd_driver.out)')
    ap.add_argument('--sgdir', default=None, help='folder of the station .sg files (for naming)')
    ap.add_argument('--glbdir', default=os.path.join(HERE, 'out_glb'))
    ap.add_argument('--stem', default='iea_s%02d.sg.glb', help='per-node .glb filename fmt (1-based node)')
    ap.add_argument('--validate-barurc', action='store_true')
    a = ap.parse_args()
    if a.validate_barurc:
        validate_barurc()
    elif a.bd:
        gen_all(a.bd, a.sgdir, a.glbdir, a.stem)
    else:
        ap.error('give --bd <beamdyn.out> or --validate-barurc')


if __name__ == '__main__':
    main()
