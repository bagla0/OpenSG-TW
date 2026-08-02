# render_fields.py -- pvpython: render the OpenSG-RM vtk AND the Abaqus
# solid vtk through the IDENTICAL pipeline: Abaqus-style 12-band rainbow
# colorbar, true isometric view (camera along (1,1,1), y up), white
# background.  Usage: pvpython render_fields.py <vtk> <outdir> <prefix>
import sys
from paraview.simple import (LegacyVTKReader, GetActiveViewOrCreate, Show,
                             ColorBy, GetColorTransferFunction,
                             GetScalarBar, Render, SaveScreenshot,
                             GetActiveCamera, ResetCamera,
                             CellDatatoPointData)

src = LegacyVTKReader(FileNames=[sys.argv[1]])
out, pref = sys.argv[2], sys.argv[3]
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 1000]
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
# the Abaqus solid vtk stores stresses as CELL data -> convert for the
# same smooth nodal contours as the RM vtk
conv = CellDatatoPointData(Input=src)
conv.ProcessAllArrays = 1
disp = Show(conv, view)
disp.Representation = 'Surface With Edges'
disp.EdgeColor = [0.25, 0.25, 0.25]


def iso_view():
    # FRONT view: face-on to the 2-D plane (looking down z, x right, y up)
    ResetCamera(view)
    cam = GetActiveCamera()
    fp = cam.GetFocalPoint()
    d = cam.GetDistance()
    cam.SetPosition(fp[0], fp[1], fp[2] + d)
    cam.SetViewUp(0, 1, 0)
    ResetCamera(view)


def shot(arrname, comp, fname, title):
    if comp is None:
        ColorBy(disp, ('POINTS', arrname))
    else:
        ColorBy(disp, ('POINTS', arrname, comp))
    lut = GetColorTransferFunction(arrname)
    lut.ApplyPreset('Blue to Red Rainbow', True)   # Abaqus spectrum
    lut.Discretize = 1
    lut.NumberOfTableValues = 12                   # Abaqus's 12 bands
    disp.RescaleTransferFunctionToDataRange(True, False)
    bar = GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ''
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    disp.SetScalarBarVisibility(view, True)
    iso_view()
    Render(view)
    SaveScreenshot(out + '/%s_%s.png' % (pref, fname), view)
    disp.SetScalarBarVisibility(view, False)


for comp in ("S11", "S22", "S33", "S12", "S13", "S23"):
    shot(comp, None, comp, comp + ' [Pa]')
for comp, ax in (("U1", 'X'), ("U2", 'Y'), ("U3", 'Z')):
    shot('disp', ax, comp, comp + ' [m]')
print('done')
