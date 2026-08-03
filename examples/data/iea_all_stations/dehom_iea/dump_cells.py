import json, glob
paths = [
 "/home/roger/a/bagla0/home/OpenSG-main/OpenSG-main/akshat_examples/Beam Model/Dehom_Timo.ipynb",
 "/home/roger/a/bagla0/home/OpenSG-main/OpenSG-main/akshat_examples/3D Model/3DModel.ipynb",
]
for path in paths:
    try:
        nb = json.load(open(path))
    except Exception as e:
        print('SKIP', path, e); continue
    print('\n' + '#' * 100)
    print('NOTEBOOK', path, '  code cells:', sum(1 for c in nb['cells'] if c['cell_type'] == 'code'))
    for ci, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        low = src.lower()
        if any(k in low for k in ('.glb', 'globalresponse', 'vabs', 'wiener', 'milenkovic',
                                  'beamdyn', '.out', 'fxr', 'fxl', 'w2p', 'rootfx')):
            print('\n----- CELL %d -----' % ci)
            print(src[:3500])
