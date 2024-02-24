import datetime
import os

import numpy as np

from utils.dimensionality_reduction import dimensionality_reduction
from utils.distance_functions import getdistancelist, calculate_element_histograms
from utils.plot_dimensionality_reduction import plot_dimensionality_reduction
from utils.read_molecule import read_molecule, getminmaxlist
from utils.remove_redundant_data import remove_redundant_data


def DV_LAE(functionpath, reffunctionpath=None, intervelnum=10, mode='tsne', inputpath=None, savename=None, refnum=1,
           num=-1, distancemode=0, interval=0.5, max_points_per_grid=1, output=None):
    if reffunctionpath is None:
        reffunctionpath = functionpath
    if refnum < 0:
        refmolecule, _ = read_molecule(reffunctionpath, index=refnum)
        _, total = read_molecule(functionpath, index=refnum)
        if isinstance(num, int) or isinstance(num, float):
            if num < 0:
                num = [total + num + 1]
        elif isinstance(num, list):
            pass
    else:
        refmolecule = read_molecule(reffunctionpath, index=refnum)
        total = read_molecule(functionpath, index=0)
        if isinstance(num, int) or isinstance(num, float):
            if num < 0:
                num = [total + num + 1]
        elif isinstance(num, list):
            pass

    maxvalue, minvalue, elementlist = getminmaxlist(functionpath)
    intevellist = {}
    for ele in elementlist:
        intevellist[ele] = (maxvalue[ele] - minvalue[ele]) / intervelnum
    refglist = calculate_element_histograms(refmolecule, maxvalue, minvalue, intevellist, elementlist, intervelnum)

    distancelist = getdistancelist(refglist, functionpath, maxvalue, minvalue, intevellist, elementlist, total,
                                   intervelnum, distancemode)

    data_2d = dimensionality_reduction(distancelist, mode)

    now = datetime.datetime.now()
    folder_path = os.path.dirname(functionpath)
    filename = os.path.basename(functionpath).split('.')[0]
    np.save(
        os.path.join(folder_path,
                     f'distancelist_b_{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}'),
        distancelist)
    data_2d_path = os.path.join(folder_path,
                                f'data_2d_b_{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}.npy')
    np.save(data_2d_path, data_2d)
    plot_dimensionality_reduction(data_2d, inputpath, num, mode, savename, functionpath, intervelnum, distancemode)
    if inputpath is not None:
        remove_redundant_data(data_2d_path, inputpath, interval, max_points_per_grid, output)
    # pass
    return data_2d, data_2d_path


if __name__ == '__main__':
    functionpath = r'example/function.data'
    inputpath = r'example/input.data'
    interval = 0.05,
    data_2d, data_2d_path = DV_LAE(functionpath, reffunctionpath=None, intervelnum=10, mode='tsne', inputpath=inputpath,
                                   savename=None, refnum=1, num=-1, distancemode=0, interval=interval)
    # remove_redundant_data(data_2d_path, inputpath, interval)
