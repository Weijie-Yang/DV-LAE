import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


class Molecule:
    def __init__(self, num_atoms):
        self.num_atoms = num_atoms
        self.atoms = []
        self.index = None

    def add_atom(self, atom_number, g_value):
        self.atoms.append((atom_number, g_value))

    def get_atomic_number_vector(self):
        atomic_number_vector = []
        for atom_number, _ in self.atoms:
            atomic_number_vector.append(atom_number)
        return atomic_number_vector

    def get_atomic_element(self):
        atomic_number_vector = []
        for atom_number, _ in self.atoms:
            atomic_number_vector.append(atom_number)
        return list(set(atomic_number_vector))

    def get_feature_vector(self):
        feature_vector = []
        for _, g_value in self.atoms:
            feature_vector.extend(g_value)
        return feature_vector

    def get_feature_matrix(self):
        feature_matrix = []
        for element, g_value in self.atoms:
            feature_matrix.append(g_value)
        return feature_matrix

    def get_feature_matrix_by_element(self, ele):
        feature_matrix = []
        for element, g_value in self.atoms:
            if ele == element:
                feature_matrix.append(g_value)
        return np.array(feature_matrix)

def read_molecule(filename, index=1):
    if index == 0:
        index = 99999999
    index1 = 0
    if index >= 0:
        with open(filename, 'r', encoding='utf-8') as file:
            while True:
                if index - 1 == index1:
                    line = file.readline().strip()
                    if not line or line == '\x1a':
                        return index1
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
                    return molecule  # 如果feature_vector已存在，将当前分子添加到对应的列表中
                else:
                    line = file.readline().strip()
                    if not line or line == '\x1a':
                        return index1
                    if line.startswith('#'):  # 忽略以 '#' 开头的行
                        continue
                    num_atoms = int(line)
                    index1 = index1 + 1
                    for _ in range(num_atoms):
                        file.readline()
                    file.readline()
    else:
        last_n_structures = []  # 用于存储最后n个结构的列表
        with open(filename, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline().strip()
                if not line or line == '\x1a':
                    return last_n_structures[0], index1
                if line.startswith('#'):  # 忽略以 '#' 开头的行
                    continue
                num_atoms = int(line)
                index1 = index1 + 1
                molecule = Molecule(num_atoms)
                molecule.index1 = index1
                for _ in range(num_atoms):
                    atom_info = file.readline().strip().split()
                    atom_number = int(atom_info[0])
                    g_value = list(map(float, atom_info[1:]))
                    molecule.add_atom(atom_number, g_value)
                file.readline()

                # 维护一个长度为n的滑动窗口，记录最后n个结构
                if len(last_n_structures) == abs(index):
                    last_n_structures.pop(0)  # 删除最早的结构，保持滑动窗口长度为n
                last_n_structures.append(molecule)

def read_n2p2(filename='output.data', index=':', with_energy_and_forces='auto'):
    fd = open(filename, 'r')  # @reader decorator ensures this is a file descriptor???
    images = list()
    lineindexlist = []
    lineindex = 0
    line = fd.readline()
    lineindex += 1
    while 'begin' in line:
        lineindexlist.append(lineindex)
        line = fd.readline()
        lineindex += 1
        if 'comment' in line:
            comment = line[7:]
            line = fd.readline()
            lineindex += 1

        cell = np.zeros((3, 3))
        for ii in range(3):
            cell[ii] = [float(jj) for jj in line.split()[1:4]]
            line = fd.readline()
            lineindex += 1

        positions = []
        symbols = []
        charges = []  # not used yet
        nn = []  # not used
        forces = []
        energy = 0.0
        charge = 0.0

        while 'atom' in line:
            sline = line.split()
            positions.append([float(pos) for pos in sline[1:4]])
            symbols.append(sline[4])
            nn.append(float(sline[5]))
            charges.append(float(sline[6]))
            forces.append([float(pos) for pos in sline[7:10]])
            line = fd.readline()
            lineindex += 1

        while 'end' not in line:
            if 'energy' in line:
                energy = float(line.split()[-1])
            if 'charge' in line:
                charge = float(line.split()[-1])
            line = fd.readline()
            lineindex += 1

        image = Atoms(symbols=symbols, positions=positions, cell=cell)

        store_energy_and_forces = False
        if with_energy_and_forces == True:
            store_energy_and_forces = True
        elif with_energy_and_forces == 'auto':
            if energy != 0.0 or np.absolute(forces).sum() > 1e-8:
                store_energy_and_forces = True

        if store_energy_and_forces:
            image.calc = SinglePointCalculator(
                atoms=image,
                energy=energy,
                forces=forces,
                charges=charges)
            # charge  = charge)
        images.append(image)
        # to start the next section
        line = fd.readline()
        lineindex += 1

    if index == ':' or index is None:
        return images, lineindexlist
    else:
        return images[index], lineindexlist

def getminmaxlist(functionpath):
    max_values = {}
    min_values = {}

    index1 = 0

    with open(functionpath, 'r') as file:
        elementlist = []
        while True:
            line = file.readline().strip()
            if not line or line == '\x1a':
                return max_values, min_values, elementlist
            if line.startswith('#'):  # 忽略以 '#' 开头的行
                continue
            num_atoms = int(line)

            glist2 = {}
            keys = []
            for _ in range(num_atoms):
                atom_info = file.readline().strip().split()
                g_value = list(map(float, atom_info[1:]))
                key = int(atom_info[0])
                keys.append(key)
                if key not in glist2:
                    glist2[key] = []
                glist2[key].append(g_value)
            if len(list(set(keys))) > len(elementlist):
                elementlist = list(set(keys))
                flag = True
            for ele in elementlist:
                if ele in glist2:
                    glist2[ele] = np.array(glist2[ele]).T
                    # 求每一行的最大值
                    max_values1 = np.amax(glist2[ele], axis=1)
                    # 求每一行的最小值
                    min_values1 = np.amin(glist2[ele], axis=1)

                    if flag or ele not in max_values:
                        max_values[ele] = max_values1
                        min_values[ele] = min_values1
                        flag = False
                    else:
                        max_values[ele] = np.maximum(max_values[ele], max_values1)
                        min_values[ele] = np.minimum(min_values[ele], min_values1)

            file.readline()

            index1 += 1