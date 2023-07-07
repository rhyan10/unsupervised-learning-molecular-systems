import py3Dmol

def view_molecule(data, index, style):

    bohr2ang = 0.529177249
    symbols = {1:'H', 6:'C', 7:'N', 8:'O', 16: 'S'}

    idx_nonzero = data['Z'][index].nonzero()
    Z = data['Z'][index][idx_nonzero]
    n_atoms = Z.size
    labels = np.vectorize(symbols.get)(Z)
    labels = labels.reshape(-1,1)

    coords = data['R'][index][0:n_atoms,:].reshape(-1,3) * bohr2ang

    xyz = np.concatenate((labels, coords), axis=1)
    n_atoms = xyz.shape[0]
    xyz_str = [str(i).strip('[]') for i in xyz]
    geom = str(n_atoms) + '\n' + ' ' + '\n'
    geom += '\n'.join(xyz_str)
    geom = geom.replace("'", "")

    for k in style.keys():
        assert k in ('line', 'stick', 'sphere', 'carton')

    molview = py3Dmol.view(width=350,height=350)
    molview.addModel(geom,'xyz')

    molview.setStyle(style)
    molview.setBackgroundColor('0xeeeeee')
    molview.zoomTo()

    return molview
