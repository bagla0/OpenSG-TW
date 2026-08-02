# odb_rpt_ex2.py -- Abaqus/CAE noGUI, the Ex.2 field exports:
#  (1) the 3-D benchmark (ex2_SOLID_field.odb, DIRECT dt=50us): clean
#      front-view contour images (white bg, mesh kept, global frame,
#      colorbar only) for U1-U3 + S11..S23, AND the raw field as .rpt:
#      S over ALL Gauss points + U at all nodes, at the 0.75 ms frame;
#  (2) the conventional FSDT shell (ex2_FSDT_step.odb): the same clean
#      images for EVERYTHING it can show -- U1-U3 and the in-plane
#      S11/S22/S12 at its stored section points -- plus its S/U rpt.
#      The point: the FSDT odb has NO s13/s23 field and NO s33 at all;
#      those images simply cannot be produced from it.
# Run:  abaqus cae noGUI=odb_rpt_ex2.py
from abaqus import *
from abaqusConstants import *
import visualization  # noqa: F401

vp = session.viewports['Viewport: 1']
session.pngOptions.setValues(imageSize=(1400, 1000))
session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)
session.graphicsOptions.setValues(backgroundStyle=SOLID,
                                  backgroundColor='#FFFFFF')
vp.viewportAnnotationOptions.setValues(triad=OFF, legend=ON, title=OFF,
                                       state=OFF, annotations=OFF,
                                       compass=OFF)
TSTAR = 0.00075


def prep(odbname):
    odb = session.openOdb(odbname)
    vp.setValues(displayedObject=odb)      # REQUIRED before writeFieldReport
    frames = odb.steps[list(odb.steps.keys())[-1]].frames
    it = min(range(len(frames)),
             key=lambda i: abs(frames[i].frameValue - TSTAR))
    vp.odbDisplay.setFrame(step=len(odb.steps) - 1, frame=it)
    vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
    vp.odbDisplay.commonOptions.setValues(visibleEdges=EXTERIOR)
    vp.view.setValues(session.views['Front'])
    vp.view.fitView()
    print('%s: frame %d at t=%.5f' % (odbname, it, frames[it].frameValue))
    return odb, it, frames


def shoot(name, comps_u, comps_s):
    for comp in comps_u:
        vp.odbDisplay.setPrimaryVariable(variableLabel='U',
                                         outputPosition=NODAL,
                                         refinement=(COMPONENT, comp))
        session.printToFile(fileName='%s_%s' % (name, comp), format=PNG,
                            canvasObjects=(vp,))
    for comp in comps_s:
        vp.odbDisplay.setPrimaryVariable(
            variableLabel='S', outputPosition=INTEGRATION_POINT,
            refinement=(COMPONENT, comp))
        session.printToFile(fileName='%s_%s' % (name, comp), format=PNG,
                            canvasObjects=(vp,))


# ---- (1) the 3-D benchmark --------------------------------------------
try:
    odb, it, frames = prep('ex2_SOLID_field.odb')
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
    shoot('ex2_SOLID_field',
          ("U1", "U2", "U3"),
          ("S11", "S22", "S33", "S12", "S13", "S23"))
    stepidx = len(odb.steps) - 1
    session.writeFieldReport(
        fileName='ex2_solid_S.rpt', append=OFF, sortItem='Element Label',
        odb=odb, step=stepidx, frame=it,
        outputPosition=INTEGRATION_POINT,
        variable=(('S', INTEGRATION_POINT),))
    session.writeFieldReport(
        fileName='ex2_solid_U.rpt', append=OFF, sortItem='Node Label',
        odb=odb, step=stepidx, frame=it, outputPosition=NODAL,
        variable=(('U', NODAL),))
    print('SOLID images + rpt done')
    odb.close()
except Exception as exc:
    print('FAIL solid: %s' % exc)

# ---- (2) the conventional FSDT shell ----------------------------------
try:
    odb, it, frames = prep('ex2_FSDT_step.odb')
    vp.odbDisplay.basicOptions.setValues(transformationType=DEFAULT)
    shoot('ex2_FSDT_step', ("U1", "U2", "U3"), ("S11", "S22", "S12"))
    stepidx = len(odb.steps) - 1
    session.writeFieldReport(
        fileName='ex2_fsdt_S.rpt', append=OFF, sortItem='Element Label',
        odb=odb, step=stepidx, frame=it,
        outputPosition=INTEGRATION_POINT,
        variable=(('S', INTEGRATION_POINT),))
    session.writeFieldReport(
        fileName='ex2_fsdt_U.rpt', append=OFF, sortItem='Node Label',
        odb=odb, step=stepidx, frame=it, outputPosition=NODAL,
        variable=(('U', NODAL),))
    print('FSDT images + rpt done (no S13/S23/S33 exist in a shell odb)')
    odb.close()
except Exception as exc:
    print('FAIL fsdt: %s' % exc)
print('all done')
