# utils.py
"""
General utility functions used across the project.
"""

import os
import glob
import numpy as np
from typing import List, Optional

def find_latest_file(pattern: str) -> Optional[str]:
    """
    Finds the most recently modified file matching a glob pattern.

    Args:
        pattern (str): The glob pattern (e.g., 'data/*.html').

    Returns:
        Optional[str]: The path to the latest file, or None if no files match.
    """
    files = glob.glob(pattern)
    if not files:
        return None
    try:
        # Find file with the maximum modification time
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    except FileNotFoundError:
        # Can happen in rare race conditions if file is deleted between glob and getmtime
        return None
    except Exception as e:
        print(f"Error finding latest file for pattern '{pattern}': {e}")
        return None


def split_list_by_values(data_list: List[int], split_points: List[int]) -> List[List[int]]:
    """
    Splits a list of numbers into sublists based on specified split points.
    Numbers less than or equal to a split point go into the corresponding list.

    Example: split_list_by_values([1, 5, 2, 8, 10, 3], [3, 8])
             -> [[1, 2, 3], [5, 8], [10]]

    Args:
        data_list (List[int]): The list of numbers to split.
        split_points (List[int]): A sorted list of values to split at.

    Returns:
        List[List[int]]: A list of sublists.
    """
    if not data_list:
        return []
    if not split_points:
        return [data_list] # Return the original list as a single sublist

    sorted_split_points = sorted(split_points)
    result = []
    remaining_items = sorted(data_list) # Sort data for easier processing
    current_sublist = []

    split_idx = 0
    current_split_value = sorted_split_points[split_idx]

    for item in remaining_items:
        # If item exceeds current split point, finalize the current sublist
        # and move to the next split point if available
        while item > current_split_value:
             result.append(current_sublist)
             current_sublist = []
             split_idx += 1
             if split_idx < len(sorted_split_points):
                 current_split_value = sorted_split_points[split_idx]
             else:
                  # This item belongs to the last sublist (greater than all split points)
                  # Set split_value effectively to infinity
                  current_split_value = float('inf')
                  # Break the inner while loop, the item will be added below

        # Add item to the correct sublist (it's <= current_split_value)
        current_sublist.append(item)

    # Add the last collected sublist
    result.append(current_sublist)

    # If there were more split points than data segments, add empty lists
    while len(result) < len(sorted_split_points) + 1:
         result.append([])

    # Original code had slightly different logic, potentially removing items.
    # This version ensures all items are placed into buckets defined by split points.
    # Re-check the original 'split_list' function's exact behavior if critical.
    # The original seemed to modify the list in place which is generally avoided.

    return result


def delete_structures_from_n2p2(filepath: str, indices_to_delete: List[int], output_suffix: str = "_new") -> Optional[str]:
    """
    Reads an n2p2 data file, removes structures at specified indices,
    and writes the result to a new file.

    Args:
        filepath (str): Path to the input n2p2 file (e.g., 'input.data').
        indices_to_delete (List[int]): A list of 0-based indices of structures to remove.
        output_suffix (str): Suffix to append to the original filename for the output file.

    Returns:
        Optional[str]: The path to the newly created file, or None if an error occurred.
    """
    print(f"Attempting to remove structures at indices {indices_to_delete} from {filepath}")
    structures = []
    current_structure_lines = []
    in_structure = False
    structure_count = 0

    try:
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith("begin"):
                    if in_structure:
                        # Found 'begin' before 'end', format error?
                        print(f"Warning: Found 'begin' before 'end' for structure ending near line {f.tell()}.")
                        # Decide how to handle: maybe save previous block?
                        if current_structure_lines:
                             structures.append("".join(current_structure_lines))
                             structure_count += 1
                    in_structure = True
                    current_structure_lines = [line] # Start new structure
                elif stripped_line.startswith("end"):
                    if not in_structure:
                        # Found 'end' without 'begin', format error?
                        print(f"Warning: Found 'end' without matching 'begin' near line {f.tell()}.")
                        # Ignore this end line or handle as error
                    else:
                        current_structure_lines.append(line)
                        structures.append("".join(current_structure_lines))
                        structure_count += 1
                        in_structure = False
                        current_structure_lines = [] # Reset for next block
                elif in_structure:
                    current_structure_lines.append(line)
                # else: line is outside begin/end block, ignore? Or preserve?
                # Current logic assumes only content within begin/end matters.

            # Check if the file ended while inside a structure block
            if in_structure and current_structure_lines:
                print(f"Warning: File ended unexpectedly within a structure block (structure {structure_count}).")
                # Decide whether to include this partial structure
                # structures.append("".join(current_structure_lines))
                # structure_count += 1

    except FileNotFoundError:
        print(f"Error: Input file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    print(f"Read {len(structures)} structures.")
    if not structures:
        print("No structures found in the file.")
        return None

    # Perform deletion using numpy's delete for potentially easier handling of indices
    try:
        indices_set = set(indices_to_delete) # For efficient lookup
        # Filter structures to keep
        structures_to_keep = [struct for i, struct in enumerate(structures) if i not in indices_set]
        num_deleted = len(structures) - len(structures_to_keep)
        print(f"Removed {num_deleted} structures.")
        # Validate if expected number deleted matches provided list length (minus duplicates/out of range)
        valid_delete_indices = {idx for idx in indices_to_delete if 0 <= idx < len(structures)}
        if num_deleted != len(valid_delete_indices):
             print(f"Warning: Requested deletion of {len(indices_to_delete)} indices, "
                   f"but actually removed {num_deleted} structures (check for duplicates or out-of-range indices).")

    except Exception as e:
        print(f"Error processing structures for deletion: {e}")
        return None

    # Define output path
    directory = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    name, ext = os.path.splitext(base_name)
    new_filename = f"{name}{output_suffix}{ext}"
    new_filepath = os.path.join(directory, new_filename)

    # Write the remaining structures to the new file
    try:
        with open(new_filepath, "w", encoding='utf-8') as f:
            for structure_block in structures_to_keep:
                f.write(structure_block)
        print(f"New file saved successfully: {new_filepath}")
        return new_filepath
    except Exception as e:
        print(f"Error writing output file {new_filepath}: {e}")
        return None