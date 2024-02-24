import os
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from collections import defaultdict

from utils.read_molecule import read_n2p2
from utils.split_list import split_list


def plot_dimensionality_reduction(data_2d, inputpath=None, num=None, mode=None, savename=None, functionpath=None,
                                  intervelnum=None, distancemode=None):
    gfig = go.Figure()
    x, y = data_2d[:, 0], data_2d[:, 1]

    if inputpath:
        atomslist, lineindexlist = read_n2p2(filename=inputpath, index=':', with_energy_and_forces='auto')
        index_dict = defaultdict(list)
        for i, atoms in enumerate(atomslist):
            index_dict[str(atoms.symbols)].append((i))

        colors = []
        for i in range(len(index_dict)):
            red = (i * 40) % 256
            green = (i * 70) % 256
            blue = (i * 100) % 256
            alpha = 0.8
            color = f'rgba({red}, {green}, {blue}, {alpha})'
            colors.append(color)

        pltlysymbols = ['circle', 'square', 'triangle-up', 'triangle-down', 'star', 'x', 'cross', 'dot']
        pltlysymbolsize = [8, 8, 8, 8, 8, 8, 8, 8]

        for i, (system, indices) in enumerate(index_dict.items()):
            color = colors[i]

            indices_lt_num = split_list(indices, num)
            for k, indices_lt in enumerate(indices_lt_num):
                gfig.add_trace(go.Scatter(
                    x=x[indices_lt],
                    y=y[indices_lt],
                    mode='markers',
                    marker=dict(color=color, symbol=pltlysymbols[k], size=pltlysymbolsize[k])
                ))

    else:
        gfig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
        ))

    gfig.update_layout(
        title=f'{mode} Visualization of Multiple Systems',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend_title='Categories'
    )

    if savename:
        plot(gfig, filename=f'{savename}.html')
    else:
        now = datetime.datetime.now()
        folder_path = os.path.dirname(functionpath)
        filename = os.path.basename(functionpath).split('.')[0]
        file_name = f'{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}.html'
        plot(gfig, filename=os.path.join(folder_path, file_name))
        np.save(os.path.join(folder_path,
                             f'index_dict_{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}'),
                index_dict)
