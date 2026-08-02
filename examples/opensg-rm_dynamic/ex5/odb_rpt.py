# odb_rpt.py -- Abaqus/CAE noGUI, two deliverables:
#  (1) contour images of the 3-D benchmarks: WHITE background, mesh grid
#      KEPT, FRONT view (face-on to the 2-D plane), global-frame stresses,
#      colorbar only (no text blocks);
#  (2) the raw field of the sandwich solid at its peak frame as .rpt:
#      S over ALL Gauss (integration) points + U at all nodes.
#      rpt_to_vtk.py (in the repo) converts the rpt into the VTK that is
#      compared against the OpenSG-RM field through the same ParaView
#      pipeline.
# Run:  abaqus cae noGUI=odb_rpt.py
from abaqus import *
from abaqusConstants import *
import visualization  # noqa: F401

# the FIELD variant runs DIRECT dt=50us with frames every 57 increments,
# so the frame nearest 0.00285 is EXACTLY the RM dump instant
CASES = [("sandwich_SOLID_field", 0.00285)]
UCOMPS = ["U1", "U2", "U3"]
SCOMPS = ["S11", "S22", "S33", "S12", "S13", "S23"]

vp = session.viewports['Viewport: 1']
session.pngOptions.setValues(imageSize=(1400, 1000))
session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)
session.graphicsOptions.setValues(backgroundStyle=SOLID,
                                  backgroundColor='#FFFFFF')
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
        vp.odbDisplay.commonOptions.setValues(visibleEdges=EXTERIOR)
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
        vp.view.setValues(session.views['Front'])  # face-on 2-D plane
        vp.view.fitView()
        for comp in UCOMPS:
            vp.odbDisplay.setPrimaryVariable(variableLabel='U',
                                             outputPosition=NODAL,
                                             refinement=(COMPONENT, comp))
            session.printToFile(fileName='%s_%s' % (name, comp),
                                format=PNG, canvasObjects=(vp,))
        for comp in SCOMPS:
            vp.odbDisplay.setPrimaryVariable(
                variableLabel='S', outputPosition=INTEGRATION_POINT,
                refinement=(COMPONENT, comp))
            session.printToFile(fileName='%s_%s' % (name, comp),
                                format=PNG, canvasObjects=(vp,))
        print('IMAGES done for %s (frame %d, t=%.5f)'
              % (name, it, frames[it].frameValue))
        if name == "sandwich_SOLID_field":
            stepidx = len(odb.steps) - 1
            session.writeFieldReport(
                fileName='sandwich_solid_step_S.rpt', append=OFF,
                sortItem='Element Label', odb=odb, step=stepidx, frame=it,
                outputPosition=INTEGRATION_POINT,
                variable=(('S', INTEGRATION_POINT),))
            session.writeFieldReport(
                fileName='sandwich_solid_step_U.rpt', append=OFF,
                sortItem='Node Label', odb=odb, step=stepidx, frame=it,
                outputPosition=NODAL, variable=(('U', NODAL),))
            print('RPT written (frame t=%.5f)' % frames[it].frameValue)
    except Exception as exc:
        print('FAIL %s: %s' % (name, exc))
    finally:
        odb.close()
print('all done')
