# molecule_structure.py
"""
Defines the Molecule class to store atomic information and feature vectors
for a single structure read from the function data file.
"""

import numpy as np
from typing import List, Tuple, Union, Set

class Molecule:
    """
    Represents a single molecular structure with its atoms and associated feature vectors (G-values).

    Attributes:
        num_atoms (int): The number of atoms in the molecule.
        atoms (List[Tuple[int, List[float]]]): A list where each tuple contains
                                               the atomic number (int) and its
                                               corresponding feature vector (List[float]).
        index (int | None): The 0-based index of the molecule within the source file.
    """
    def __init__(self, num_atoms: int):
        """
        Initializes a Molecule instance.

        Args:
            num_atoms (int): The number of atoms this molecule will contain.
        """
        if not isinstance(num_atoms, int) or num_atoms <= 0:
            raise ValueError("num_atoms must be a positive integer.")
        self.num_atoms: int = num_atoms
        self.atoms: List[Tuple[int, List[float]]] = []
        self.index: Union[int, None] = None # Index in the file

    def add_atom(self, atom_number: int, g_value: List[float]):
        """
        Adds an atom and its feature vector to the molecule.

        Args:
            atom_number (int): The atomic number of the atom (e.g., 1 for H, 6 for C).
            g_value (List[float]): The list of feature values (G-values) for this atom.
        """
        if not isinstance(atom_number, int) or atom_number <= 0:
            raise ValueError("atom_number must be a positive integer.")
        if not isinstance(g_value, list) or not all(isinstance(g, (int, float)) for g in g_value):
             raise ValueError("g_value must be a list of numbers.")
        if len(self.atoms) >= self.num_atoms:
            raise IndexError("Cannot add more atoms than specified by num_atoms.")

        self.atoms.append((atom_number, g_value))

    def get_atomic_number_vector(self) -> List[int]:
        """
        Returns a list of atomic numbers for all atoms in the molecule.

        Returns:
            List[int]: A list containing the atomic number of each atom in order.
        """
        return [atom_number for atom_number, _ in self.atoms]

    def get_atomic_elements(self) -> Set[int]:
        """
        Returns a set of unique atomic numbers (elements) present in the molecule.

        Returns:
            Set[int]: A set containing the unique atomic numbers.
        """
        return set(self.get_atomic_number_vector())

    def get_feature_vector(self) -> List[float]:
        """
        DEPRECATED (potentially misleading): Returns a flattened list of all feature values
        from all atoms. The order depends on the atom order. Use get_feature_matrix
        or get_feature_matrix_by_element for more structured data.

        Returns:
            List[float]: A single list containing all feature values concatenated.
        """
        # Warning: This flattens features across atoms, losing structure.
        # Consider if this representation is truly desired.
        feature_vector = []
        for _, g_value in self.atoms:
            feature_vector.extend(g_value)
        return feature_vector

    def get_feature_matrix(self) -> List[List[float]]:
        """
        Returns a list of feature vectors, one for each atom.

        Returns:
            List[List[float]]: A list where each inner list is the feature vector for an atom.
                               The outer list follows the order of atoms added.
        """
        return [g_value for _, g_value in self.atoms]

    def get_feature_matrix_by_element(self, element_atomic_number: int) -> np.ndarray:
        """
        Returns a NumPy array containing the feature vectors for all atoms
        of a specific element.

        Args:
            element_atomic_number (int): The atomic number of the element to filter by.

        Returns:
            np.ndarray: A 2D NumPy array where each row is the feature vector
                        of an atom matching the specified element. Returns an empty
                        array with the correct shape if the element is not found.
        """
        feature_matrix = [g_value for atom_number, g_value in self.atoms if atom_number == element_atomic_number]
        # Determine expected feature dimension length from the first atom if possible
        feature_dim = 0
        if self.atoms:
            feature_dim = len(self.atoms[0][1])
        # Return empty array with correct shape if no atoms of this element exist
        return np.array(feature_matrix) if feature_matrix else np.empty((0, feature_dim))

    def __len__(self) -> int:
        """Returns the number of atoms added so far."""
        return len(self.atoms)

    def __str__(self) -> str:
        """String representation of the Molecule."""
        elements = sorted(list(self.get_atomic_elements()))
        return f"<Molecule index={self.index} num_atoms={self.num_atoms} elements={elements}>"

    def __repr__(self) -> str:
        """Detailed representation of the Molecule."""
        return self.__str__()