# render_fields.py -- pvpython: render the OpenSG-RM vtk AND the Abaqus
# solid vtk through the IDENTICAL pipeline: front view, Abaqus-style
# 12-band rainbow colorbar, white background, mesh edges, and the model
# TITLE on every image.  One PNG per component (6 stresses + 3 disp) into
# the given output folder.
# Usage: pvpython render_fields.py <vtk> <outdir> <title>
import os
import sys
from paraview.simple import (LegacyVTKReader, GetActiveViewOrCreate, Show,
                             ColorBy, GetColorTransferFunction,
                             GetScalarBar, Render, SaveScreenshot,
                             GetActiveCamera, ResetCamera,
                             CellDatatoPointData, Text)

src = LegacyVTKReader(FileNames=[sys.argv[1]])
out, model_title = sys.argv[2], sys.argv[3]
if not os.path.isdir(out):
    os.makedirs(out)
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
# the model title on every image
txt = Text()
txt.Text = model_title
tdisp = Show(txt, view)
tdisp.WindowLocation = 'Upper Center'
tdisp.FontSize = 22
tdisp.Color = [0, 0, 0]
tdisp.Bold = 1


def set_view(mode):
    # 'front': face-on to the 2-D plane (looking down z, x right, y up);
    # 'sidex': the x = 0 edge face (thickness vertical) -- where s13/s33
    #          live; 'sidey': the y = 0 edge face -- where s23 lives
    ResetCamera(view)
    cam = GetActiveCamera()
    fp = cam.GetFocalPoint()
    d = cam.GetDistance()
    if mode == 'front':
        cam.SetPosition(fp[0], fp[1], fp[2] + d)
        cam.SetViewUp(0, 1, 0)
    elif mode == 'sidex':
        cam.SetPosition(fp[0] - d, fp[1], fp[2])
        cam.SetViewUp(0, 0, 1)
    else:
        cam.SetPosition(fp[0], fp[1] - d, fp[2])
        cam.SetViewUp(0, 0, 1)
    ResetCamera(view)


VIEWS = {"S13": "sidex", "S33": "sidex", "S23": "sidey", "S22": "sidex"}
# filename tag: the case and the snapshot instant, e.g. ex5_step_t2p85ms
TAG = sys.argv[4] if len(sys.argv) > 4 else ""


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
    set_view(VIEWS.get(fname, 'front'))
    Render(view)
    stem = fname + ('_' + TAG if TAG else '')
    SaveScreenshot(out + '/%s.png' % stem, view)
    disp.SetScalarBarVisibility(view, False)


for comp in ("S11", "S22", "S33", "S12", "S13", "S23"):
    shot(comp, None, comp, comp + ' [Pa]')
for comp, ax in (("U1", 'X'), ("U2", 'Y'), ("U3", 'Z')):
    shot('disp', ax, comp, comp + ' [m]')
print('done')
