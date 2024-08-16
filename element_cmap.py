import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

def prep_phase_colormap():
    """
    Generate discrete colormap with black bottom (invalidated points).

    Returns
    -------
    phase_cmap : colormap

    """

    cmap = mpl.pyplot.get_cmap('Set1', 9)
    phase_cmap = mpl.pyplot.get_cmap('Set1', 9)
    phase_cmap.colors[0] = [0,0,0,1]
    phase_cmap.colors[1:] = cmap.colors[:8]
    
    return phase_cmap


def prep_elemental_colormaps():
    """
    Prepare and register EDAX APEX style colormaps for each element, which can be called as e.g. "cmap_Ti".

    Returns
    -------
    None.

    """
    
    element_colors = {
        'H' :'#e6e6fa',
        'He':'#0000ff',
        'Li':'#e6e6fa',
        'Be':'#e6e6fa',
        'B' :'#ff9700',
        'C' :'#800080',
        'N' :'#ffc0cb',
        'O' :'#7fff00',
        'F' :'#ffa500',
        'Ne':'#0000ff',
        'Na':'#cd853f',
        'Mg':'#a52a2a',
        'Al':'#add8e6',
        'Si':'#ee82ee',
        'P' :'#00ff00',
        'S' :'#ba55d3',
        'Cl':'#ffc0cb',
        'Ar':'#0000ff',
        'K' :'#2f4f4f',
        'Ca':'#008b8b',
        'Sc':'#f0e68c',
        'Ti':'#5f9ea0',
        'V' :'#f08080',
        'Cr':'#ffe4c4',
        'Mn':'#ba55d3',
        'Fe':'#ff7f50',
        'Co':'#87cefa',
        'Ni':'#ffa500',
        'Cu':'#9acd32',
        'Zn':'#2e8b57',
        'Ga':'#00ffff',
        'Ge':'#dda0dd',
        'As':'#c71585',
        'Se':'#800080',
        'Br':'#add8e6',
        'Kr':'#0000ff',
        'Rb':'#ffc0cb',
        'Sr':'#d8bfd8',
        'Y' :'#bdb76b',
        'Zr':'#ff6347',
        'Nb':'#daa520',
        'Mo':'#ff00ff',
        'Tc':'#fff8dc',
        'Ru':'#ff69b4',
        'Rh':'#dcdcdc',
        'Pd':'#c0c0c0',
        'Ag':'#ff7f50',
        'Cd':'#f5deb3',
        'In':'#708090',
        'Sn':'#90ee90',
        'Sb':'#40e0d0',
        'Te':'#b8860b',
        'I' :'#808000',
        'Xe':'#0000ff',
        'Cs':'#ffdab9',
        'Ba':'#add8e6',
        'La':'#ffff00',
        'Ce':'#ffff00',
        'Pr':'#ffff00',
        'Nd':'#ffff00',
        'Pm':'#ffff00',
        'Sm':'#ffff00',
        'Eu':'#ffff00',
        'Gd':'#ffff00',
        'Tb':'#ffff00',
        'Dy':'#ffff00',
        'Ho':'#ffff00',
        'Er':'#ffff00',
        'Tm':'#ffff00',
        'Yb':'#ffff00',
        'Lu':'#ffff00',
        'Hf':'#bc8f8f',
        'Ta':'#87cefa',
        'W' :'#ff4500',
        'Re':'#d8bfd8',
        'Os':'#a0522d',
        'Ir':'#e9967a',
        'Pt':'#7fff00',
        'Au':'#f4a460',
        'Hg':'#d2b48c',
        'Tl':'#008080',
        'Pb':'#d2b48c',
        'Bi':'#b22222',
        'Po':'#4682b4',
        'At':'#d2691e',
        'Rn':'#0000ff',
        'Fr':'#dda0dd',
        'Ra':'#4169e1',
        'Ac':'#fffacd',
        'Th':'#ffff00',
        'Pa':'#ffff00',
        'U' :'#ffff00',
        'Np':'#ffff00',
        'Pu':'#ffff00',
        'Am':'#ffff00',
        'Cm':'#ffff00',
        'Bk':'#ffff00',
        'Cf':'#ffff00',
        'Es':'#ffff00',
        'Fm':'#ffff00',
        'Md':'#ffff00',
        'No':'#ffff00',
        'Lr':'#ffff00',
        'Rf':'#ffff00',
        'Db':'#ffff00',
        'Sg':'#ffff00',
        'Bh':'#ffff00',
        'Hs':'#ffff00',
        'Mt':'#ffff00',
        'Ds':'#ffff00',
        'Rg':'#ffff00',
        'Cn':'#ffff00',
        'Nh':'#ffff00',
        'Fl':'#ffff00',
        'Mc':'#ffff00',
        'Lv':'#ffff00',
        'Ts':'#ffff00',
        'Og':'#ffff00'
    }
    
    cmap_element_list = list()
    
    for key in element_colors:
        mpl.colormaps.register(LinearSegmentedColormap.from_list("cmap_" + key, ['#000000', element_colors[key]]))
        cmap_element_list.append("cmap_" + key)
