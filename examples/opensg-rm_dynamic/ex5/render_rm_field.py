# render_rm_field.py -- pvpython: contour the OpenSG-RM field vtk over the
# 3-D geometry: rainbow (Abaqus-style blue->red) colorbar, Abaqus-like iso
# view, no mesh lines.  One PNG per component (6 stresses + 3 disp).
import sys
from paraview.simple import (LegacyVTKReader, GetActiveViewOrCreate, Show,
                             ColorBy, GetColorTransferFunction,
                             GetScalarBar, Render, SaveScreenshot,
                             GetActiveCamera, ResetCamera)

src = LegacyVTKReader(FileNames=[sys.argv[1]])
out = sys.argv[2]
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 1000]
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
disp = Show(src, view)
disp.Representation = 'Surface'


def abaqus_view():
    ResetCamera(view)
    cam = GetActiveCamera()
    cam.Azimuth(35)
    cam.Elevation(22)
    ResetCamera(view)


def shot(arrname, comp, fname, title):
    if comp is None:
        ColorBy(disp, ('POINTS', arrname))
        lut = GetColorTransferFunction(arrname)
    else:
        ColorBy(disp, ('POINTS', arrname, comp))
        lut = GetColorTransferFunction(arrname)
    lut.ApplyPreset('Blue to Red Rainbow', True)   # the Abaqus spectrum
    disp.RescaleTransferFunctionToDataRange(True, False)
    bar = GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ''
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    disp.SetScalarBarVisibility(view, True)
    abaqus_view()
    Render(view)
    SaveScreenshot(out + '/rm_field_%s.png' % fname, view)
    disp.SetScalarBarVisibility(view, False)


for comp in ("S11", "S22", "S33", "S12", "S13", "S23"):
    shot(comp, None, comp, comp + ' [Pa]')
for comp, ax in (("U1", 'X'), ("U2", 'Y'), ("U3", 'Z')):
    shot('disp', ax, comp, comp + ' [m]')
print('done')
