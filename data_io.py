# data_io.py
"""
Functions for reading molecular structures and features from different file formats.
Supports reading feature data (function.data format) and atomic structures
(n2p2 output.data format, or standard formats via ASE).
"""
import os

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from typing import List, Tuple, Union, Optional, Dict, TextIO

# Import the Molecule class from its module
try:
    from molecule_structure import Molecule
except ImportError:
    # Fallback for environments where the module structure isn't set up
    # This allows individual file execution but isn't ideal for package structure
    print("Warning: Could not import Molecule from molecule_structure. Assuming definition exists.")
    # Define a minimal placeholder if needed for linting/basic checks
    class Molecule: pass


def read_feature_file_structure(file_handle: TextIO, current_index: int) -> Union[Molecule, None]:
    """
    Reads a single structure block from an open feature file (function.data format).

    Args:
        file_handle (TextIO): An open file object positioned at the start of a structure block.
        current_index (int): The 0-based index for this structure.

    Returns:
        Molecule | None: A Molecule object if a structure is successfully read,
                        None if the end of the file or an invalid format is encountered.
    """
    while True: # Skip comments and blank lines before the atom count
        line = file_handle.readline()
        if not line: # End of file
            return None
        line = line.strip()
        if not line or line == '\x1a': # Empty line or EOF character
             return None
        if line.startswith('#'): # Skip comment lines
             continue
        # Found the line with the number of atoms
        break

    try:
        num_atoms = int(line)
        if num_atoms <= 0:
             print(f"Warning: Structure index {current_index} reports non-positive atom count ({num_atoms}). Skipping.")
             # Attempt to skip the expected number of lines plus the potential extra line
             for _ in range(num_atoms + 1): file_handle.readline()
             return None # Indicate failure to read this structure
    except ValueError:
        print(f"Warning: Could not parse number of atoms at index {current_index}. Line: '{line}'. Skipping.")
        # Cannot reliably skip, might corrupt reading next structure. Return None.
        return None

    molecule = Molecule(num_atoms)
    molecule.index = current_index

    try:
        for i in range(num_atoms):
            atom_line = file_handle.readline()
            if not atom_line:
                 print(f"Warning: Unexpected end of file while reading atoms for structure index {current_index}.")
                 return None # Incomplete structure
            atom_info = atom_line.strip().split()
            if len(atom_info) < 2:
                 print(f"Warning: Malformed atom line {i+1} for structure index {current_index}. Line: '{atom_line.strip()}'")
                 # Attempt to continue, but this structure might be corrupt
                 continue # Or return None for stricter parsing

            atom_number = int(atom_info[0])
            g_value = list(map(float, atom_info[1:]))
            molecule.add_atom(atom_number, g_value)

        # Skip the extra line often present between structures in .data files
        file_handle.readline()

        if len(molecule) != num_atoms:
             print(f"Warning: Read {len(molecule)} atoms but expected {num_atoms} for structure index {current_index}.")
             # Decide whether to return the partially read molecule or None
             # return None # Stricter approach
             pass # Allow partially read molecule for now

        return molecule

    except (ValueError, IndexError) as e:
        print(f"Error reading atoms for structure index {current_index}: {e}")
        return None # Failed to read this structure properly


def read_single_molecule_features(filename: str, index: int = 1) -> Union[Molecule, int, Tuple[Molecule, int]]:
    """
    Reads a single Molecule object with features from a specified index in a file.

    Args:
        filename (str): Path to the feature data file (e.g., 'function.data').
        index (int): The 1-based index of the structure to read.
                     If 0, reads until the end to count structures.
                     If negative (-n), reads the nth structure from the end.

    Returns:
        Molecule: If a positive index is given and found.
        int: If index is 0, returns the total count of structures found.
        Tuple[Molecule, int]: If a negative index is given, returns the requested
                               Molecule and the total count of structures.
        Raises FileNotFoundError if the file cannot be opened.
        Raises ValueError for invalid index.
    """
    if not isinstance(index, int):
        raise ValueError("Index must be an integer.")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            current_index = 0
            if index > 0:
                # Read specific structure by 1-based index
                target_0_index = index - 1
                while True:
                    molecule = read_feature_file_structure(f, current_index)
                    if molecule is None: # EOF or error
                        raise IndexError(f"Structure index {index} not found. File contains {current_index} structures.")
                    if current_index == target_0_index:
                        return molecule
                    current_index += 1
            elif index == 0:
                # Count structures
                while True:
                     molecule = read_feature_file_structure(f, current_index)
                     if molecule is None: # EOF or error reading last structure
                         return current_index # Return count of successfully started structures
                     current_index += 1
            else: # Negative index
                # Read all structures, keeping the last N in memory
                n = abs(index)
                last_n_structures: List[Molecule] = []
                while True:
                    molecule = read_feature_file_structure(f, current_index)
                    if molecule is None: # EOF
                        if not last_n_structures:
                             raise IndexError(f"Could not read any structures to find the {n}th from end.")
                        if n > len(last_n_structures):
                             raise IndexError(f"Requested {n}th structure from end, but only {len(last_n_structures)} found.")
                        return last_n_structures[-n], current_index # Return molecule and total count
                    last_n_structures.append(molecule)
                    if len(last_n_structures) > n:
                        last_n_structures.pop(0) # Keep only the last n
                    current_index += 1

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading {filename}: {e}")
        raise


def read_n2p2_output(filename: str = 'output.data', index: Union[str, int, slice] = ':', with_energy_and_forces: Union[bool, str] = 'auto') -> Union[List[Atoms], Atoms, Tuple[List[Atoms], List[int]]]:
    """
    Reads atomic structures from an n2p2 output data file (like output.data).
    Closely based on ASE's internal n2p2 reader logic.

    Args:
        filename (str): Path to the n2p2 output file.
        index (Union[str, int, slice]): Specifies which structures to return.
            ':' or None returns all images as a list.
            An integer returns a single Atoms object.
            A slice object returns a list of Atoms objects.
        with_energy_and_forces (Union[bool, str]): Whether to attach energy and forces.
            True: Always attach if present in the file.
            False: Never attach.
            'auto': Attach only if energy is non-zero or forces are significant.

    Returns:
        Union[List[Atoms], Atoms, Tuple[List[Atoms], List[int]]]:
            Depending on the index, returns a single Atoms object or a list of them.
            If index is ':', also returns a list of the starting line numbers for each frame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is unexpected.
    """
    images = []
    line_indices = [] # Store starting line number (1-based) of each 'begin' block

    try:
        with open(filename, 'r', encoding='utf-8') as fd:
            line_number = 0
            while True:
                line = fd.readline()
                if not line:
                    break # End of file
                line_number += 1

                if 'begin' in line:
                    line_indices.append(line_number)
                    # Start reading a structure block
                    comment = ""
                    cell = np.zeros((3, 3))
                    positions = []
                    symbols = []
                    charges_read = [] # Use dedicated list for reading charges if present
                    forces = []
                    energy = 0.0
                    total_charge = 0.0 # Overall system charge

                    # Read comment if present
                    line = fd.readline().strip()
                    line_number += 1
                    if line.startswith('comment'):
                        comment = line[7:].strip()
                        line = fd.readline().strip()
                        line_number += 1
                    else:
                         # No comment line, the current line should be the first lattice vector
                         pass # line variable already holds the potential lattice line

                    # Read lattice vectors
                    try:
                        for i in range(3):
                            if not line.startswith('lattice'):
                                raise ValueError(f"Expected 'lattice' at line {line_number}, got: {line}")
                            cell[i] = [float(x) for x in line.split()[1:4]]
                            line = fd.readline().strip()
                            line_number += 1
                    except (IndexError, ValueError) as e:
                         raise ValueError(f"Error parsing lattice vectors near line {line_number}: {e}")

                    # Read atoms
                    try:
                        while line.startswith('atom'):
                            sline = line.split()
                            # Format: atom x y z symbol element_charge_?? element_charge_?? Fx Fy Fz (optional_other_stuff)
                            if len(sline) < 10:
                                raise ValueError(f"Atom line too short at line {line_number}: {line}")
                            positions.append([float(pos) for pos in sline[1:4]])
                            symbols.append(sline[4])
                            # N2P2 format might include charges per atom here (sline[5], sline[6]?)
                            # Assuming charges are read later or defaulted to 0 for now.
                            charges_read.append(0.0) # Placeholder
                            forces.append([float(f) for f in sline[7:10]])
                            line = fd.readline().strip()
                            line_number += 1
                    except (IndexError, ValueError) as e:
                         raise ValueError(f"Error parsing atom line near line {line_number}: {e}")

                    # Read energy, charge, and end marker
                    while 'end' not in line:
                        if not line: # Premature end of file
                            raise ValueError(f"Unexpected end of file before 'end' marker near line {line_number}")
                        if line.startswith('energy'):
                            try:
                                energy = float(line.split()[-1])
                            except (IndexError, ValueError):
                                 raise ValueError(f"Could not parse energy at line {line_number}: {line}")
                        elif line.startswith('charge'):
                             try:
                                 # Note: Original code hardcoded charge=0. Reading if present.
                                 total_charge = float(line.split()[-1])
                             except (IndexError, ValueError):
                                  raise ValueError(f"Could not parse charge at line {line_number}: {line}")
                        # Can add reading other properties here if needed
                        line = fd.readline()
                        if not line: # Check again after reading
                            raise ValueError(f"Unexpected end of file before 'end' marker after line {line_number}")
                        line = line.strip()
                        line_number += 1

                    # Create ASE Atoms object
                    # N2P2 often doesn't enforce pbc=True in the file, but it usually implies periodic cells
                    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

                    # Apply sorting by atomic number as in original code (optional, but maintains consistency)
                    sorted_indices = np.argsort(atoms.numbers)
                    atoms = atoms[sorted_indices]
                    # Reorder forces and potentially read charges according to sorted indices
                    forces = np.array(forces)[sorted_indices]
                    # If charges_read were actually populated, they need sorting too:
                    # charges_read = np.array(charges_read)[sorted_indices]

                    # Decide whether to store energy/forces
                    store_ef = False
                    if isinstance(with_energy_and_forces, bool):
                        store_ef = with_energy_and_forces
                    elif with_energy_and_forces == 'auto':
                        # Check if energy exists or forces are significant
                        if abs(energy) > 1e-12 or np.sum(np.abs(forces)) > 1e-8:
                            store_ef = True
                    else:
                        raise ValueError(f"Invalid value for with_energy_and_forces: {with_energy_and_forces}")

                    if store_ef:
                        # ASE's SinglePointCalculator doesn't directly store total charge in a standard way.
                        # We can store per-atom charges if available (using charges_read).
                        # The total charge read from the file (`total_charge`) is stored as an attribute for now.
                        atoms.calc = SinglePointCalculator(
                            atoms=atoms,
                            energy=energy,
                            forces=forces,
                            # charges=charges_read # Use if per-atom charges are read
                        )
                        # Store total charge as an extra attribute if needed
                        # atoms.info['total_charge'] = total_charge # Example

                    if comment:
                         atoms.info['comment'] = comment

                    images.append(atoms)

                # Continue reading until next 'begin' or EOF

    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        raise

    # Return based on index
    if index == ':' or index is None:
        return images, line_indices
    elif isinstance(index, (int, slice)):
        # Standard Python list slicing/indexing handles single int or slice
        try:
            return images[index]
        except IndexError:
            raise IndexError(f"Index {index} out of range for {len(images)} structures in {filename}")
    else:
        raise TypeError(f"Unsupported index type: {type(index)}")


def read_atomic_structures(filename: str, index: Union[str, int, slice] = ':') -> Union[List[Atoms], Atoms]:
    """
    Reads atomic structures from various file formats using ASE.
    Provides a generic interface, attempting n2p2 format first if suffix is '.data'.

    Args:
        filename (str): Path to the structure file (e.g., input.data, trajectory.xyz).
        index (Union[str, int, slice]): Specifies which structures to return (ASE format).

    Returns:
        Union[List[Atoms], Atoms]: A single Atoms object or a list of them.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        Exception: Various ASE exceptions depending on file format issues.
    """
    if not filename:
         raise ValueError("Input filename must be provided.")

    file_extension = os.path.splitext(filename)[1].lower()

    try:
        if file_extension == '.data':
            # Try n2p2 format first for .data files, assuming it won't have energy/forces
            # unless explicitly requested or needed later. Return only atoms list.
            try:
                 images, _ = read_n2p2_output(filename=filename, index=':', with_energy_and_forces=False)
                 if index == ':' or index is None:
                     return images
                 else:
                     return images[index]
            except ValueError as e:
                 print(f"Reading {filename} as n2p2 failed ({e}), falling back to generic ASE reader.")
                 # Fall through to generic ASE read if n2p2 parsing fails
            except Exception as e:
                 print(f"Unexpected error reading {filename} as n2p2 ({e}), falling back to generic ASE reader.")
                 # Fall through

        # Use generic ASE reader for other formats or as fallback
        atoms_list = ase_read(filename, index=index)
        return atoms_list

    except FileNotFoundError:
        print(f"Error: Structure file not found at {filename}")
        raise
    except Exception as e:
        print(f"Error reading structure file {filename} using ASE: {e}")
        raise