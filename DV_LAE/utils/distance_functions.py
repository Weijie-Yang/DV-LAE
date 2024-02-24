import numpy as np
from tqdm import tqdm

from utils.read_molecule import Molecule


def calculate_distance3(hist1, hist2):
    # 当二者范围不等时取1否则取0
    result = [0 if (x == y and x == 0) or (x != 0 and y != 0) else 1 for x, y in zip(hist1, hist2)]
    # 当二者有不同时则取1，否则取0
    return sum(result)

def calculate_distance2(hist1, hist2):
    return np.sqrt(np.sum((hist1 - hist2) ** 2))


def calculate_distance1(hist1, hist2):
    # 当二者范围不等时取1否则取0
    result = [0 if (x == y and x == 0) or (x != 0 and y != 0) else 1 for x, y in zip(hist1, hist2)]
    # 当二者有不同时则取1，否则取0
    return 1 if sum(result) > 1 else 0
def calculate_element_histograms(molecule, maxvalue, minvalue, intevellist, elementlist, intervelnum):
    elements = molecule.get_atomic_element()
    gdic = {}
    for ele in elementlist:
        if ele in elements:
            matrix_by_element = (molecule.get_feature_matrix_by_element(ele)).T
            gdic[ele] = []
            for i, e in enumerate(matrix_by_element):
                # 计算整数区间的边界，包含最大最小值
                if intevellist[ele][i] != 0:
                    hist, bin_edges = np.histogram(e,
                                                   bins=np.arange(minvalue[ele][i],
                                                                  maxvalue[ele][i] + intevellist[ele][i],
                                                                  intevellist[ele][i]))
                else:
                    hist = np.array([len(e)] + [0] * (intervelnum - 1))
                    bin_edges = np.arange(np.unique(e)[0], np.unique(e)[0] + 15, 1)
                gdic[ele].append((hist, bin_edges))
        else:
            gdic[ele] = []
            for i in range(len(intevellist[ele])):
                if intevellist[ele][i] != 0:
                    hist, bin_edges = np.histogram(np.array([]),
                                                   bins=np.arange(minvalue[ele][i],
                                                                  maxvalue[ele][i] + intevellist[ele][i],
                                                                  intevellist[ele][i]))
                else:
                    hist = np.array([len(e)] + [0] * (intervelnum - 1))
                    bin_edges = np.arange(np.unique(e)[0], np.unique(e)[0] + 15, 1)
                gdic[ele].append((hist, bin_edges))
            pass

    return gdic

def calculate_element_histogram_distances(glist1, glist2, elementlist, distancemode=0):
    distancelist = {}

    for ele in elementlist:
        distancelist[ele] = []

        for i in range(len(glist1[ele])):
            hist1 = glist1[ele][i][0]
            hist2 = glist2[ele][i][0]

            if distancemode == 0:
                distance = calculate_distance1(hist1, hist2)
            elif distancemode == 1:
                distance = calculate_distance2(hist1, hist2)
            elif distancemode == 2:
                distance = calculate_distance3(hist1, hist2)
            else:
                raise ValueError("Invalid distancemode value")

            distancelist[ele].append(distance)

    return distancelist

def getdistancelist(refglist, functionpath, maxvalue, minvalue, intevellist, elementlist, total, intervelnum,
                    distancemode):
    jiangweilist = []
    index1 = 0
    progress_bar = tqdm(range(total))
    with open(functionpath, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line or line == '\x1a':
                progress_bar.close()
                return jiangweilist
            if line.startswith('#'):  # 忽略以 '#' 开头的行
                continue
            num_atoms = int(line)
            molecule = Molecule(num_atoms)
            molecule.index1 = index1

            for _ in range(num_atoms):
                atom_info = file.readline().strip().split()
                atom_number = int(atom_info[0])
                g_value = list(map(float, atom_info[1:]))
                molecule.add_atom(atom_number, g_value)
            file.readline()
            # distancelist.append(functionname3(refglist, functionname1(molecule, intervel=intervel),
            #                                   intervel))
            m = calculate_element_histogram_distances(refglist,
                                                      calculate_element_histograms(molecule, maxvalue, minvalue,
                                                                                   intevellist, elementlist,
                                                                                   intervelnum),
                                                      elementlist, distancemode=distancemode)

            jiangweilist.append([])
            for k in elementlist:
                jiangweilist[index1] = np.concatenate((jiangweilist[index1], m[k]))

            index1 += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Progress": index1 / total})