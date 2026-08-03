r'''run_vabs_51.py  --  drive VABS over all 51 IEA cross-sections (RUN ON THE WINDOWS MACHINE where
VABS is licensed;  Y:\ maps to the server home, so the server-side .sg / .glb are visible directly).

Two VABS calls per station (exactly the syntax the user asked for):
  1. HOMOGENIZE :  vabs iea_sNN.sg          -> writes iea_sNN.sg.K   (Timoshenko 6x6 + mass, VABS gold)
  2. RECOVER    :  vabs iea_sNN.sg 2 0      -> reads  iea_sNN.sg.glb  (global response from BeamDyn),
                                              writes the 3-D stress/strain recovery (.U*, .S* fields)

VABS reads the global-response file as "<inputfile>.glb", i.e. iea_sNN.sg -> iea_sNN.sg.glb, and it
must sit BESIDE the .sg.  The .glb come from gen_glb_from_beamdyn.py (--glbdir glb51); this script
copies each glb51/iea_sNN.sg.glb next to its iea_sNN.sg before the recovery call.

Stations s02 and s50 have no PreVABS .sg (thin-wall / tip-sliver PreVABS failures -> meshed by the
pyNuMAD-quad path instead), so VABS is simply skipped there and reported.

    python run_vabs_51.py                       # homogenize + recover all stations that have a .sg
    python run_vabs_51.py --step homo           # homogenize only  (vabs iea_sNN.sg)
    python run_vabs_51.py --step dehom          # recover only     (vabs iea_sNN.sg 2 0)
    python run_vabs_51.py --dry-run             # print the commands without running VABS
    python run_vabs_51.py --vabs "C:\path\VABS.exe"   # explicit VABS executable
'''
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SGDIR_DEFAULT = os.path.normpath(os.path.join(HERE, '..', 'shell51', 'sg_v201'))
GLBDIR_DEFAULT = os.path.join(HERE, 'glb51')
NSTA = 51


def run(cmd, cwd, dry):
    print('   $ %s   (cwd=%s)' % (' '.join(cmd), cwd))
    if dry:
        return 0, ''
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if p.stdout.strip():
            print('     ' + p.stdout.strip().replace('\n', '\n     '))
        if p.returncode != 0 and p.stderr.strip():
            print('     [stderr] ' + p.stderr.strip().replace('\n', '\n     '))
        return p.returncode, p.stdout
    except FileNotFoundError:
        print('     !! VABS executable not found: %s' % cmd[0])
        return 127, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sgdir', default=SGDIR_DEFAULT, help='folder of iea_sNN.sg files')
    ap.add_argument('--glbdir', default=GLBDIR_DEFAULT, help='folder of iea_sNN.sg.glb files')
    ap.add_argument('--vabs', default='vabs', help='VABS executable (on PATH or full path)')
    ap.add_argument('--step', choices=['homo', 'dehom', 'both'], default='both')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    print('VABS driver over %d IEA stations' % NSTA)
    print('  sg  dir : %s' % a.sgdir)
    print('  glb dir : %s' % a.glbdir)
    print('  vabs    : %s   step=%s%s\n' % (a.vabs, a.step, '   (DRY RUN)' if a.dry_run else ''))

    done, skipped, failed = [], [], []
    for i in range(NSTA):
        sgname = 'iea_s%02d.sg' % i
        sgpath = os.path.join(a.sgdir, sgname)
        if not os.path.isfile(sgpath):
            skipped.append(i)
            print('s%02d : no .sg (PreVABS mesh failure -> pyNuMAD-quad path); VABS skipped' % i)
            continue
        print('s%02d : %s' % (i, sgname))
        rc = 0

        # 1. homogenization:  vabs iea_sNN.sg
        if a.step in ('homo', 'both'):
            rc, _ = run([a.vabs, sgname], a.sgdir, a.dry_run)

        # 2. recovery:  stage the .glb beside the .sg, then  vabs iea_sNN.sg 2 0
        if a.step in ('dehom', 'both') and rc == 0:
            glbsrc = os.path.join(a.glbdir, sgname + '.glb')            # iea_sNN.sg.glb
            glbdst = os.path.join(a.sgdir, sgname + '.glb')
            if not os.path.isfile(glbsrc):
                print('     !! missing %s ; run gen_glb_from_beamdyn.py first' % glbsrc)
                failed.append(i); continue
            if os.path.abspath(glbsrc) != os.path.abspath(glbdst) and not a.dry_run:
                shutil.copyfile(glbsrc, glbdst)
            rc, _ = run([a.vabs, sgname, '2', '0'], a.sgdir, a.dry_run)

        (done if rc == 0 else failed).append(i)

    print('\n---- summary ----')
    print('  ran     : %d stations %s' % (len(done), ['s%02d' % i for i in done]))
    print('  skipped : %d stations %s  (no .sg)' % (len(skipped), ['s%02d' % i for i in skipped]))
    if failed:
        print('  FAILED  : %d stations %s' % (len(failed), ['s%02d' % i for i in failed]))
    if a.dry_run:
        print('\n(dry run -- no VABS was executed)')


if __name__ == '__main__':
    sys.exit(main())
