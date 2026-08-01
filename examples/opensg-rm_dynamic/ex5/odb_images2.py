# odb_images2.py -- Abaqus/CAE noGUI: CLEAN contour images from the 3-D
# BENCHMARK odbs only: image + colorbar, no state/title text, no compass,
# feature edges only (no mesh grid).  3 displacement + 6 stress components
# per case, stresses transformed to the GLOBAL frame.  Plus one image of
# the applied double-sine pressure load over the geometry (from the .inp).
# Run:  abaqus cae noGUI=odb_images2.py
from abaqus import *
from abaqusConstants import *
import visualization  # noqa: F401

CASES = [("sandwich_SOLID_step", 0.00285), ("ex2_SOLID_step", 0.00218)]
UCOMPS = ["U1", "U2", "U3"]
SCOMPS = ["S11", "S22", "S33", "S12", "S13", "S23"]

vp = session.viewports['Viewport: 1']
session.pngOptions.setValues(imageSize=(1400, 1000))
session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)
# no text blocks: keep ONLY the legend (colorbar)
vp.viewportAnnotationOptions.setValues(triad=OFF, legend=ON, title=OFF,
                                       state=OFF, annotations=OFF,
                                       compass=OFF)
for name, tstar in CASES:
    try:
        odb = session.openOdb(name + '.odb')
    except Exception as exc:
        print('SKIP %s: %s' % (name, exc))
        continue
    try:
        vp.setValues(displayedObject=odb)
        frames = odb.steps[list(odb.steps.keys())[-1]].frames
        it = min(range(len(frames)),
                 key=lambda i: abs(frames[i].frameValue - tstar))
        vp.odbDisplay.setFrame(step=len(odb.steps) - 1, frame=it)
        vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
        # no mesh grid: outline/feature edges only
        vp.odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)
        # stresses in the GLOBAL frame (the plies carry *ORIENTATION):
        # the standard scratch-odb global-CSYS transformation recipe
        try:
            scr = session.ScratchOdb(odb)
        except Exception:
            scr = session.scratchOdbs[odb.name]
        try:
            scr.rootAssembly.DatumCsysByThreePoints(
                name='GLOB', coordSysType=CARTESIAN, origin=(0, 0, 0),
                point1=(1, 0, 0), point2=(0, 1, 0))
        except Exception:
            pass
        vp.odbDisplay.basicOptions.setValues(
            transformationType=USER_SPECIFIED,
            datumCsys=scr.rootAssembly.datumCsyses['GLOB'])
        vp.view.setValues(session.views['Iso'])
        vp.view.fitView()
        for comp in UCOMPS:
            vp.odbDisplay.setPrimaryVariable(variableLabel='U',
                                             outputPosition=NODAL,
                                             refinement=(COMPONENT, comp))
            session.printToFile(fileName='%s_%s' % (name, comp),
                                format=PNG, canvasObjects=(vp,))
            print('WROTE %s_%s.png (t=%.5f)' % (name, comp,
                                                frames[it].frameValue))
        for comp in SCOMPS:
            vp.odbDisplay.setPrimaryVariable(
                variableLabel='S', outputPosition=INTEGRATION_POINT,
                refinement=(COMPONENT, comp))
            session.printToFile(fileName='%s_%s' % (name, comp),
                                format=PNG, canvasObjects=(vp,))
            print('WROTE %s_%s.png' % (name, comp))
    except Exception as exc:
        print('FAIL %s: %s' % (name, exc))
    finally:
        odb.close()

# ---- the applied load over the geometry (from the solid .inp) -----------
try:
    mdb.ModelFromInputFile(name='LOADPIC',
                           inputFileName='sandwich_SOLID_step.inp')
    a = mdb.models['LOADPIC'].rootAssembly
    vp.setValues(displayedObject=a)
    vp.assemblyDisplay.setValues(loads=ON, bcs=OFF, mesh=ON,
                                 step='PULSE')
    vp.assemblyDisplay.meshOptions.setValues(meshVisibleEdges=FEATURE)
    vp.view.setValues(session.views['Iso'])
    vp.view.fitView()
    session.printToFile(fileName='sandwich_applied_load', format=PNG,
                        canvasObjects=(vp,))
    print('WROTE sandwich_applied_load.png')
except Exception as exc:
    print('FAIL loadpic: %s' % exc)
print('all done')
