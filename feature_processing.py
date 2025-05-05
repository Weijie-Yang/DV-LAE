# feature_processing.py
"""
Functions for processing feature vectors (G-values) from Molecule objects.
Includes calculating histograms, finding min/max ranges, and computing
distance vectors between structures based on their feature histograms.
"""

import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from typing import Union
# Import necessary classes/functions from other modules
try:
    from molecule_structure import Molecule
    from data_io import read_feature_file_structure # Used in get_feature_ranges
except ImportError:
    print("Warning: Could not import Molecule/read_feature_file_structure. Assuming definitions exist.")
    class Molecule: pass
    def read_feature_file_structure(*args, **kwargs): pass


# --- Distance Metric Calculation Helpers ---

def _hist_range_difference(hist1: np.ndarray, hist2: np.ndarray) -> List[int]:
    """
    Compares two histograms bin-wise. Returns 1 if bins represent different
    ranges (one has counts, the other doesn't, excluding the case where both are empty),
    otherwise 0.

    Args:
        hist1 (np.ndarray): First histogram count array.
        hist2 (np.ndarray): Second histogram count array.

    Returns:
        List[int]: List of 0s and 1s indicating range disparity for each bin.
    """
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same number of bins for comparison.")
    # 1 if (x > 0 and y == 0) or (x == 0 and y > 0), else 0
    # Simplified: 1 if (x == 0) != (y == 0), else 0
    return [1 if (x == 0) != (y == 0) else 0 for x, y in zip(hist1, hist2)]

def _hist_range_difference_binary(hist1: np.ndarray, hist2: np.ndarray) -> int:
    """
    Checks if there's *any* significant range difference between two histograms.
    Returns 1 if there is at least one bin where one histogram has counts and
    the other doesn't (ignoring bins where both are empty), otherwise 0.

    Args:
        hist1 (np.ndarray): First histogram count array.
        hist2 (np.ndarray): Second histogram count array.

    Returns:
        int: 1 if any range difference exists, 0 otherwise.
    """
    diff_vector = _hist_range_difference(hist1, hist2)
    # Original logic checked sum > 1, but sum > 0 seems more logical for *any* difference
    return 1 if sum(diff_vector) > 0 else 0


def _hist_elementwise_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.ndarray:
    """
    Calculates the element-wise Euclidean distance (sqrt((h1_i - h2_i)^2))
    between two histograms. Note: This is just the absolute difference.

    Args:
        hist1 (np.ndarray): First histogram count array.
        hist2 (np.ndarray): Second histogram count array.

    Returns:
        np.ndarray: Array of absolute differences between corresponding bins.
    """
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same number of bins.")
    # np.sqrt((hist1 - hist2)**2) is equivalent to np.abs(hist1 - hist2)
    return np.abs(hist1 - hist2)


# --- Core Feature Processing Functions ---

def calculate_feature_histograms(
    molecule: Molecule,
    max_values: Dict[int, np.ndarray],
    min_values: Dict[int, np.ndarray],
    intervals: Dict[int, np.ndarray],
    element_list: List[int],
    num_bins: int
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Calculates histograms for each feature dimension for each element in a molecule,
    based on pre-defined ranges and intervals.

    Args:
        molecule (Molecule): The molecule object containing feature data.
        max_values (Dict[int, np.ndarray]): Dictionary mapping element atomic number
                                             to an array of max values for each feature dim.
        min_values (Dict[int, np.ndarray]): Dictionary mapping element atomic number
                                             to an array of min values for each feature dim.
        intervals (Dict[int, np.ndarray]): Dictionary mapping element atomic number
                                           to an array of interval widths for each feature dim.
                                           A zero interval indicates a single bin.
        element_list (List[int]): The list of all unique elements across the dataset.
        num_bins (int): The target number of bins for histogram calculation when interval > 0.

    Returns:
        Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
            A dictionary where keys are element atomic numbers. Values are lists,
            where each element corresponds to a feature dimension for that element type.
            Each tuple in the list contains:
                - hist (np.ndarray): The histogram counts.
                - bin_edges (np.ndarray): The edges of the histogram bins.
    """
    present_elements = molecule.get_atomic_elements()
    hist_data = {}

    for element in element_list:
        hist_data[element] = []
        if element in present_elements:
            # Get feature matrix: rows=atoms, cols=features -> transpose to rows=features, cols=atoms
            feature_matrix = molecule.get_feature_matrix_by_element(element).T # Shape: (num_features, num_atoms_of_element)

            if feature_matrix.size == 0: # Handle case where element exists but has no features somehow?
                 # Add empty histograms based on expected dimensions from max_values/min_values
                 num_features = len(max_values.get(element, []))
                 for i in range(num_features):
                     hist = np.zeros(num_bins, dtype=int)
                     # Create dummy bin edges if possible, otherwise maybe default range?
                     min_val = min_values[element][i]
                     max_val = max_values[element][i]
                     interval = intervals[element][i]
                     if interval > 0:
                          bin_edges = np.linspace(min_val, max_val + interval, num_bins + 1) # Adjusted for consistency
                     else:
                          # Single point case needs careful definition of edges
                          unique_val = min_val # Assume min=max if interval is 0
                          bin_edges = np.array([unique_val, unique_val + 1e-9]) # Tiny range for single bin
                     hist_data[element].append((hist, bin_edges))
                 continue


            num_features = feature_matrix.shape[0]

            for i in range(num_features): # Iterate through each feature dimension
                feature_values = feature_matrix[i] # Values for this feature across all atoms of this element
                min_val = min_values[element][i]
                max_val = max_values[element][i]
                interval_width = intervals[element][i]

                if interval_width > 0:
                    # Define bin edges ensuring max value is included
                    # Using linspace might be slightly more robust for float precision
                    # Add small epsilon to max_val to ensure inclusion if data point hits exactly max_val
                    epsilon = 1e-9
                    # Need num_bins + 1 edges to define num_bins bins
                    # Range should span from min_val to max_val
                    # The original binning was slightly ambiguous with arange and interval
                    # Let's define bins based on num_bins over the range [min_val, max_val]
                    bin_edges = np.linspace(min_val, max_val + epsilon, num_bins + 1)

                    # Calculate histogram
                    hist, _ = np.histogram(feature_values, bins=bin_edges)

                    # Ensure hist has length num_bins, might need adjustment if histogram returns fewer
                    if len(hist) < num_bins:
                         # This shouldn't happen with linspace if num_bins > 0
                         padded_hist = np.zeros(num_bins, dtype=hist.dtype)
                         padded_hist[:len(hist)] = hist
                         hist = padded_hist


                else: # Zero interval width - treat as categorical or single point value
                    # All values should be identical (or very close)
                    # We want a "histogram" with one bin counting all atoms, and others zero
                    hist = np.zeros(num_bins, dtype=int)
                    if feature_values.size > 0:
                        hist[0] = len(feature_values) # Count all atoms in the first bin
                    # Define dummy bin edges around the unique value
                    unique_val = min_val # Assuming min=max if interval is 0
                    # Create edges that represent this single point bin conceptually
                    bin_edges = np.linspace(unique_val, unique_val + epsilon, num_bins + 1) # Placeholder edges


                hist_data[element].append((hist, bin_edges))

        else: # Element not present in this molecule
            # Create empty histograms for all feature dimensions of this element type
            num_features = len(max_values.get(element, [])) # Get expected feature count
            for i in range(num_features):
                hist = np.zeros(num_bins, dtype=int)
                min_val = min_values[element][i]
                max_val = max_values[element][i]
                interval_width = intervals[element][i]
                epsilon=1e-9

                # Define bin edges even for empty histograms
                if interval_width > 0:
                     bin_edges = np.linspace(min_val, max_val + epsilon, num_bins + 1)
                else:
                     unique_val = min_val
                     bin_edges = np.linspace(unique_val, unique_val + epsilon, num_bins + 1) # Placeholder

                hist_data[element].append((hist, bin_edges))

    return hist_data


def calculate_feature_distance_vector(
    hist_data1: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    hist_data2: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    element_list: List[int],
    distance_mode: int = 0
) -> Dict[int, Union[List[int], np.ndarray]]:
    """
    Calculates a distance vector between two structures based on their feature histograms.

    Args:
        hist_data1: Histogram data for the first structure (output of calculate_feature_histograms).
        hist_data2: Histogram data for the second structure.
        element_list (List[int]): List of all elements considered.
        distance_mode (int): Specifies the type of distance calculation:
            0: Binary range difference (1 if any difference per feature, 0 otherwise).
            1: Element-wise absolute difference between histogram counts (concatenated).
            2: Binary range difference per bin (concatenated).

    Returns:
        Dict[int, Union[List[int], np.ndarray]]:
            A dictionary mapping element atomic number to its distance vector component.
            The type of the vector depends on the distance_mode.
            Mode 0: List of 0s/1s (one per feature).
            Mode 1: NumPy array of absolute differences (concatenated across features).
            Mode 2: NumPy array of 0s/1s (concatenated across bins and features).
    """
    distance_vectors = {}

    for element in element_list:
        distance_vectors[element] = [] # Initialize as list, convert to array if needed

        # Ensure both structures have data for this element (even if empty histograms)
        if element not in hist_data1 or element not in hist_data2:
             print(f"Warning: Element {element} missing from histogram data of one structure.")
             # Decide how to handle this: skip, assume max distance?
             # For now, create an empty list/array of appropriate type if possible
             # This depends heavily on knowing the expected feature/bin count, which is tricky here
             # Let's assume the structure is consistent and this shouldn't happen if element_list is correct
             continue # Skip this element if data is missing

        features1 = hist_data1[element]
        features2 = hist_data2[element]

        if len(features1) != len(features2):
             print(f"Warning: Mismatch in number of features for element {element} between structures.")
             # Handle mismatch, e.g., skip element or pad? Skipping for now.
             continue

        num_features = len(features1)
        if num_features == 0:
             continue # No features for this element type

        if distance_mode == 0:
            # Binary difference per feature (0 if similar range, 1 if different)
            for i in range(num_features):
                hist1, _ = features1[i]
                hist2, _ = features2[i]
                distance_vectors[element].append(_hist_range_difference_binary(hist1, hist2))
        elif distance_mode == 1:
            # Element-wise absolute difference, concatenated
            all_diffs = []
            for i in range(num_features):
                hist1, _ = features1[i]
                hist2, _ = features2[i]
                all_diffs.extend(_hist_elementwise_distance(hist1, hist2))
            distance_vectors[element] = np.array(all_diffs)
        elif distance_mode == 2:
            # Binary difference per bin, concatenated
            all_bin_diffs = []
            for i in range(num_features):
                hist1, _ = features1[i]
                hist2, _ = features2[i]
                all_bin_diffs.extend(_hist_range_difference(hist1, hist2))
            distance_vectors[element] = np.array(all_bin_diffs)
        else:
            raise ValueError(f"Unsupported distance_mode: {distance_mode}")

    return distance_vectors


def get_feature_ranges(feature_filepath: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    """
    Reads through the entire feature file to determine the minimum and maximum
    value for each feature dimension, aggregated by element type.

    Args:
        feature_filepath (str): Path to the feature data file (e.g., 'function.data').

    Returns:
        Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
            - max_values (Dict): Maps element atomic number to array of max feature values.
            - min_values (Dict): Maps element atomic number to array of min feature values.
            - element_list (List): Sorted list of unique element atomic numbers found.
    """
    max_values = {}
    min_values = {}
    elements_found: Set[int] = set()
    first_structure_processed = False

    print(f"Calculating feature ranges from: {feature_filepath}")
    try:
        with open(feature_filepath, 'r', encoding='utf-8') as f:
            structure_index = 0
            while True:
                molecule = read_feature_file_structure(f, structure_index)
                if molecule is None:
                    break # End of file or read error

                current_elements = molecule.get_atomic_elements()
                elements_found.update(current_elements)

                # Group features by element for this molecule
                features_by_element: Dict[int, List[List[float]]] = {}
                for el in current_elements:
                    features_by_element[el] = [] # Initialize list for this element

                for atom_num, g_vec in molecule.atoms:
                    if atom_num in features_by_element:
                        features_by_element[atom_num].append(g_vec)
                    # No else needed, already initialized based on current_elements

                # Update overall min/max values
                for element, feature_list in features_by_element.items():
                    if not feature_list: continue # Skip if element had no atoms (shouldn't happen)

                    # Convert to NumPy array: shape (num_atoms_of_element, num_features)
                    feature_matrix = np.array(feature_list)
                    if feature_matrix.size == 0: continue # Skip empty feature sets

                    # Calculate min/max along the atom axis (axis=0)
                    current_max = np.amax(feature_matrix, axis=0)
                    current_min = np.amin(feature_matrix, axis=0)

                    if not first_structure_processed or element not in max_values:
                        # Initialize for this element or if it's the very first data
                        max_values[element] = current_max
                        min_values[element] = current_min
                    else:
                        # Update existing min/max
                        # Ensure dimensions match before comparison (should match if format is consistent)
                        if max_values[element].shape != current_max.shape:
                             print(f"Warning: Feature dimension mismatch for element {element} at structure {structure_index}. Skipping update.")
                             continue # Or raise error
                        max_values[element] = np.maximum(max_values[element], current_max)
                        min_values[element] = np.minimum(min_values[element], current_min)

                first_structure_processed = True # Mark that we have initial values
                structure_index += 1
                if structure_index % 500 == 0: # Progress update
                     print(f"  Processed {structure_index} structures for ranges...")

    except FileNotFoundError:
        print(f"Error: Feature file not found at {feature_filepath}")
        raise
    except Exception as e:
        print(f"An error occurred while reading feature ranges from {feature_filepath}: {e}")
        raise

    if not max_values or not min_values:
        print("Warning: No features read from file. Cannot determine ranges.")
        return {}, {}, []

    element_list = sorted(list(elements_found))
    print(f"Feature ranges calculation complete. Found elements: {element_list}")
    # Sanity check: Ensure all elements in element_list have entries in min/max dicts
    for el in element_list:
         if el not in min_values or el not in max_values:
              print(f"Warning: Element {el} was found but has no min/max values. Check file consistency.")
              # Need to decide how to handle this: remove from list, add dummy values?
              # Adding dummy zero arrays of expected shape if possible.
              # Find expected shape from another element
              expected_shape = None
              if max_values:
                  expected_shape = next(iter(max_values.values())).shape
              if expected_shape:
                    min_values[el] = np.zeros(expected_shape)
                    max_values[el] = np.zeros(expected_shape)
              else: # Cannot determine shape, remove from list
                    print(f"  Cannot determine feature shape for element {el}, removing from list.")
                    element_list.remove(el)


    return max_values, min_values, element_list


def compute_all_distance_vectors(
    feature_filepath: str,
    ref_hist_data: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    max_values: Dict[int, np.ndarray],
    min_values: Dict[int, np.ndarray],
    intervals: Dict[int, np.ndarray],
    element_list: List[int],
    total_structures: int,
    num_bins: int,
    distance_mode: int
) -> List[np.ndarray]:
    """
    Computes the distance vector (based on feature histograms) between a reference
    structure and every other structure in the feature file.

    Args:
        feature_filepath (str): Path to the feature data file.
        ref_hist_data: Histogram data for the reference structure.
        max_values, min_values, intervals: Range and interval info for histogramming.
        element_list: List of all elements.
        total_structures (int): The total number of structures expected in the file.
        num_bins (int): Number of bins used for histograms.
        distance_mode (int): The distance mode (0, 1, or 2).

    Returns:
        List[np.ndarray]: A list where each element is the concatenated distance
                          vector (across all elements) for a structure compared to the reference.
    """
    all_distance_vectors = []
    print(f"Computing distance vectors relative to reference (mode={distance_mode})...")

    try:
        with open(feature_filepath, 'r', encoding='utf-8') as f:
            # Use tqdm for progress bar
            pbar = tqdm(range(total_structures), desc="Processing structures")
            for structure_index in pbar:
                molecule = read_feature_file_structure(f, structure_index)
                if molecule is None:
                    print(f"\nWarning: Expected {total_structures} structures, but reading stopped at index {structure_index}.")
                    break # Stop if file ends prematurely or error occurs

                # 1. Calculate histograms for the current molecule
                current_hist_data = calculate_feature_histograms(
                    molecule, max_values, min_values, intervals, element_list, num_bins
                )

                # 2. Calculate the distance vector relative to the reference
                dist_vector_components = calculate_feature_distance_vector(
                    ref_hist_data, current_hist_data, element_list, distance_mode
                )

                # 3. Concatenate components into a single vector for this structure
                # Ensure consistent order using element_list
                concatenated_vector = []
                for element in element_list:
                    component = dist_vector_components.get(element, []) # Get component or empty list if missing
                    if isinstance(component, np.ndarray):
                        concatenated_vector.extend(component.tolist())
                    else: # Should be list for mode 0
                        concatenated_vector.extend(component)

                all_distance_vectors.append(np.array(concatenated_vector))

                # Update progress bar postfix if desired
                # pbar.set_postfix({"Processed": structure_index + 1})

    except FileNotFoundError:
        print(f"Error: Feature file not found at {feature_filepath}")
        raise
    except Exception as e:
        print(f"An error occurred while computing distance vectors: {e}")
        raise

    print(f"Finished computing {len(all_distance_vectors)} distance vectors.")
    return all_distance_vectors