# structure_filtering.py
"""
Filters molecular structures based on the distribution of their
corresponding points in a low-dimensional (2D) space.

This script reads 2D coordinate data (e.g., from t-SNE/PCA/UMAP),
divides the 2D space into a grid, selects a representative subset
of points (and their original indices) from this grid, and then
extracts the corresponding full atomic structures from an input
structure file (like input.data or trajectory.xyz).
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set

# Assuming ASE is installed for reading/writing various formats
try:
    from ase.io import read as ase_read, write as ase_write
except ImportError:
    print("Warning: ASE not found. Reading/writing formats other than n2p2 '.data' will fail.")
    print("Install ASE: pip install ase")
    ase_read = None
    ase_write = None

# --- Core Grid Selection Logic ---

def select_indices_from_grid(
    data_2d: np.ndarray,
    grid_interval: float,
    max_points_per_cell: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divides the 2D space into a grid and selects representative indices.

    Args:
        data_2d (np.ndarray): The 2D data points (n_samples, 2).
        grid_interval (float): The spacing for grid lines along both x and y axes.
        max_points_per_cell (int): Maximum number of points to randomly select
                                   from each non-empty grid cell.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - selected_indices (np.ndarray): Array of unique indices selected from the grid.
            - x_bins (np.ndarray): The calculated bin edges along the x-axis.
            - y_bins (np.ndarray): The calculated bin edges along the y-axis.

    Raises:
        ValueError: If data_2d is not 2-dimensional or grid_interval is non-positive.
    """
    if data_2d.ndim != 2 or data_2d.shape[1] != 2:
        raise ValueError("Input data_2d must be a 2D NumPy array with shape (n_samples, 2).")
    if grid_interval <= 0:
        raise ValueError("grid_interval must be positive.")
    if data_2d.shape[0] == 0:
        print("Warning: Input data_2d is empty. Returning empty selection.")
        return np.array([], dtype=int), np.array([]), np.array([])

    # Get data range
    x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
    y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])

    # Define grid bins - add a small epsilon to max to ensure inclusion
    epsilon = 1e-9
    x_bins = np.arange(x_min, x_max + grid_interval + epsilon, grid_interval)
    y_bins = np.arange(y_min, y_max + grid_interval + epsilon, grid_interval)

    selected_indices_list: List[int] = []
    num_x_bins = len(x_bins) - 1
    num_y_bins = len(y_bins) - 1

    print(f"Creating grid: {num_x_bins} x-bins, {num_y_bins} y-bins.")

    # Iterate through grid cells
    for i in range(num_x_bins):
        for j in range(num_y_bins):
            x_min_cell, x_max_cell = x_bins[i], x_bins[i+1]
            y_min_cell, y_max_cell = y_bins[j], y_bins[j+1]

            # Find indices of points within the current cell
            # Note: Using '<' for x_max_cell and y_max_cell to match np.histogram behavior
            #       Points exactly on the upper edge fall into the next bin.
            indices_in_cell = np.where(
                (data_2d[:, 0] >= x_min_cell) & (data_2d[:, 0] < x_max_cell) &
                (data_2d[:, 1] >= y_min_cell) & (data_2d[:, 1] < y_max_cell)
            )[0]

            num_found = len(indices_in_cell)
            if num_found > 0:
                if num_found <= max_points_per_cell:
                    # Take all points if count is within limit
                    selected_indices_in_cell = indices_in_cell
                else:
                    # Randomly sample the required number of points
                    selected_indices_in_cell = np.random.choice(
                        indices_in_cell, size=max_points_per_cell, replace=False
                    )
                selected_indices_list.extend(selected_indices_in_cell.tolist())

    # Convert to numpy array and ensure uniqueness (though sampling shouldn't duplicate)
    selected_indices = np.unique(np.array(selected_indices_list, dtype=int))

    print(f"Selected {len(selected_indices)} unique indices from the grid.")
    return selected_indices, x_bins, y_bins

# --- Plotting ---

def plot_grid_selection(
    all_points: np.ndarray,
    selected_indices: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    title: str = 'Grid-Based Point Selection'
) -> None:
    """
    Generates a scatter plot showing all points, selected points, and the grid.

    Args:
        all_points (np.ndarray): The original 2D data points (n_samples, 2).
        selected_indices (np.ndarray): Indices of the points selected by the grid algorithm.
        x_bins (np.ndarray): Grid bin edges along the x-axis.
        y_bins (np.ndarray): Grid bin edges along the y-axis.
        title (str): Title for the plot.
    """
    if not plt:
         print("Matplotlib is required for plotting but not available.")
         return
    if all_points.ndim != 2 or all_points.shape[1] != 2:
         print("Warning: Invalid shape for 'all_points' in plotting function.")
         return

    plt.figure(figsize=(10, 8))

    # Plot all points (optional, can be commented out for clarity if many points)
    # plt.scatter(all_points[:, 0], all_points[:, 1], color='lightblue', label='All Points', s=10, alpha=0.5)

    # Plot selected points
    if selected_indices.size > 0:
        selected_points = all_points[selected_indices]
        plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red', label=f'Selected Points ({len(selected_indices)})', s=20, edgecolors='k', linewidths=0.5)
    else:
        print("No points were selected to plot.")

    # Draw grid lines
    for x in x_bins:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
    for y in y_bins:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

    plt.legend()
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(False) # Disable default grid, using our custom lines
    plt.gca().set_aspect('equal', adjustable='box') # Often helps visualize density
    plt.show()

# --- Structure Filtering and Saving ---

def filter_and_save_structures(
    selected_indices: np.ndarray,
    input_structure_path: str,
    output_structure_path: str
) -> None:
    """
    Reads structures from an input file, selects those corresponding to the
    provided indices, and saves them to a new file. Handles n2p2 '.data'
    format specifically and uses ASE for other formats.

    Args:
        selected_indices (np.ndarray): 0-based indices of structures to keep.
        input_structure_path (str): Path to the original structure file.
        output_structure_path (str): Path where the filtered structures will be saved.

    Raises:
        FileNotFoundError: If the input structure file does not exist.
        ImportError: If ASE is needed but not installed.
        IOError: If there are issues reading or writing files.
    """
    if not os.path.exists(input_structure_path):
        raise FileNotFoundError(f"Input structure file not found: {input_structure_path}")

    # Use a set for efficient checking of indices to keep
    indices_to_keep: Set[int] = set(selected_indices.tolist())
    num_to_keep = len(indices_to_keep)
    if num_to_keep == 0:
        print("Warning: No indices selected. Output file will be empty.")
        # Create empty file
        with open(output_structure_path, 'w') as f_out:
            pass
        print(f"Empty output file created: {output_structure_path}")
        return

    file_extension = os.path.splitext(input_structure_path)[1].lower()
    structures_written = 0

    print(f"Filtering structures from '{os.path.basename(input_structure_path)}', keeping {num_to_keep} structures.")

    if file_extension == '.data':
        # --- Specific handling for n2p2 .data format ---
        try:
            with open(input_structure_path, 'r', newline='\n') as f_in, \
                 open(output_structure_path, 'w', newline='\n') as f_out:

                current_structure_index = 0
                in_structure_block = False
                lines_for_current_structure = []

                for line in f_in:
                    stripped_line = line.strip()

                    if stripped_line.startswith("begin"):
                        if in_structure_block:
                            # Handle unexpected 'begin' before 'end'
                            print(f"Warning: Found 'begin' before 'end' around structure index {current_structure_index}. Discarding previous partial block.")
                        in_structure_block = True
                        lines_for_current_structure = [line] # Start new block
                    elif stripped_line.startswith("end"):
                        if not in_structure_block:
                            # Handle unexpected 'end'
                            print(f"Warning: Found 'end' without matching 'begin'. Ignoring line.")
                        else:
                            lines_for_current_structure.append(line)
                            # Check if this completed structure should be kept
                            if current_structure_index in indices_to_keep:
                                f_out.writelines(lines_for_current_structure)
                                structures_written += 1
                            # Reset for next structure
                            in_structure_block = False
                            lines_for_current_structure = []
                            current_structure_index += 1 # Increment index AFTER processing 'end'
                    elif in_structure_block:
                        lines_for_current_structure.append(line)
                    # Ignore lines outside begin/end blocks

                # Check if file ended unexpectedly
                if in_structure_block:
                    print(f"Warning: Input file ended while processing structure index {current_structure_index}.")

        except IOError as e:
            raise IOError(f"Error reading/writing n2p2 .data files: {e}")

    else:
        # --- Generic handling using ASE ---
        if ase_read is None or ase_write is None:
            raise ImportError("ASE is required to handle non-.data structure files.")
        try:
            print(f"Using ASE to read '{input_structure_path}'...")
            # Read all structures - might be memory intensive for huge files
            all_atoms = ase_read(input_structure_path, index=':')
            print(f"Read {len(all_atoms)} structures using ASE.")

            # Filter the list
            selected_atoms = [atoms for i, atoms in enumerate(all_atoms) if i in indices_to_keep]
            structures_written = len(selected_atoms)

            if structures_written == 0:
                 print("Warning: No structures matched the selected indices after reading with ASE.")

            # Write selected structures
            # Infer format from output extension, default to xyz if ambiguous
            output_format = os.path.splitext(output_structure_path)[1][1:] # Get 'xyz' from '.xyz'
            if not output_format:
                 output_format = 'xyz'
                 output_structure_path += '.xyz' # Append extension if missing
                 print(f"Output format inferred as '{output_format}', saving to {output_structure_path}")

            print(f"Writing {structures_written} structures to '{output_structure_path}' (format: {output_format})...")
            ase_write(output_structure_path, selected_atoms, format=output_format)

        except Exception as e:
            # Catch potential ASE reading/writing errors
            raise IOError(f"Error during ASE structure processing: {e}")

    print(f"Successfully wrote {structures_written} structures to '{output_structure_path}'.")
    if structures_written != num_to_keep:
         print(f"Warning: Expected to write {num_to_keep} structures, but wrote {structures_written}. Check input file consistency and indices.")

# --- File Finding Utility ---

def find_coordinate_files(
    directory: str,
    prefix: str,
    number_filter: Optional[List[int]] = None,
    dist_mode_filter: Optional[List[int]] = None
) -> List[str]:
    """
    Finds .npy coordinate files in a directory matching a specific pattern.

    Pattern assumes filenames like:
    {timestamp}_{basename}_{bins}_{reduction_mode}_{dist_mode}_coords_2d.npy

    Args:
        directory (str): The directory to search in.
        prefix (str): The expected prefix (e.g., '202310271130_function'). This part
                      should combine the timestamp and original basename.
                      Alternatively, use '*' if timestamp/basename vary wildly.
        number_filter (Optional[List[int]]): If provided, only finds files where the
                                            'bins' part matches one of these numbers.
        dist_mode_filter (Optional[List[int]]): If provided, only finds files where the
                                               'dist_mode' part matches one of these numbers.

    Returns:
        List[str]: A list of matching file paths.
    """
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found: {directory}")
        return []

    # Construct glob pattern dynamically
    pattern_parts = [prefix]

    if number_filter:
        pattern_parts.append(f"{{{','.join(map(str, number_filter))}}}") # e.g., {10,20,30}
    else:
        pattern_parts.append("*") # Match any number (bins)

    pattern_parts.append("*") # Match any reduction_mode (tsne, pca, etc.)

    if dist_mode_filter:
        pattern_parts.append(f"{{{','.join(map(str, dist_mode_filter))}}}") # e.g., {0,1,2}
    else:
        pattern_parts.append("*") # Match any distance mode

    pattern_parts.append("coords_2d.npy")

    # Join parts with '_' and combine with directory
    file_pattern = os.path.join(directory, "_".join(pattern_parts))
    # Note: Depending on shell/OS, curly braces might need escaping or different handling.
    # Python's glob should handle them correctly. Let's refine if issues arise.
    # Simpler approach if filters apply to only one part at a time:
    # Example for filtering only by number:
    # file_pattern = os.path.join(directory, f"{prefix}_{number}_*_coords_2d.npy")

    # More robust glob pattern for the example format:
    # {timestamp}_{basename}_{bins}_{reduction_mode}_{dist_mode}_coords_2d.npy
    # Let's assume prefix is just the 'coords_2d' part for simplicity in finding,
    # and filter later if needed, or make the pattern more specific.
    # Let's stick to the original script's find logic which seemed more targeted:
    # find files like: {base_path}/{prefix}_*_{number}_tsne_{mode}.npy

    # Revised find function based on original script's apparent intent:
    matching_files = []
    bins_str = "*" if number_filter is None else f"{{{','.join(map(str, number_filter))}}}"
    mode_str = "*" if dist_mode_filter is None else f"{{{','.join(map(str, dist_mode_filter))}}}"

    # Assuming specific reduction mode ('tsne' hardcoded in original find logic)
    # To make it flexible, we'd need another parameter or loop
    reduction_mode_to_find = "tsne" # Or pass as argument

    file_pattern = os.path.join(directory, f"{prefix}_*_{bins_str}_{reduction_mode_to_find}_{mode_str}_coords_2d.npy")
    # Need to handle glob limitations with complex patterns like braces on some systems.
    # Let's glob broadly and filter afterwards for reliability.

    broad_pattern = os.path.join(directory, f"{prefix}_*_coords_2d.npy")
    all_coord_files = glob.glob(broad_pattern)
    print(f"Found {len(all_coord_files)} potential coordinate files matching pattern '{broad_pattern}'.")


    for fpath in all_coord_files:
        fname = os.path.basename(fpath)
        parts = fname.replace("_coords_2d.npy", "").split('_')
        # Expected parts based on example: [prefix_part1, prefix_part2, ..., bins, reduction_mode, dist_mode]
        if len(parts) < 3: continue # Needs at least bins, mode, dist_mode

        try:
            file_bins = int(parts[-3])
            file_reduction = parts[-2]
            file_dist_mode = int(parts[-1])

            # Apply filters
            bins_match = number_filter is None or file_bins in number_filter
            reduction_match = file_reduction == reduction_mode_to_find # Match specific mode
            dist_mode_match = dist_mode_filter is None or file_dist_mode in dist_mode_filter

            if bins_match and reduction_match and dist_mode_match:
                matching_files.append(fpath)

        except (ValueError, IndexError):
            # Filename doesn't match expected format
            continue

    print(f"Filtered down to {len(matching_files)} files matching criteria.")
    return matching_files


# --- Main Execution Logic ---

def main():
    """Parses arguments and runs the grid filtering workflow."""
    parser = argparse.ArgumentParser(description="Filter structures based on 2D grid sampling.")
    parser.add_argument("coord_files", nargs='+',
                        help="Path(s) to the 2D coordinate .npy file(s). "
                             "Can use wildcards like '*.npy'.")
    parser.add_argument("--input_structure", required=True,
                        help="Path to the original full structure file (e.g., input.data, trajectory.xyz).")
    parser.add_argument("--output_dir", default=".",
                        help="Directory to save the filtered output structure files (default: current dir).")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="Grid interval size (default: 0.1).")
    parser.add_argument("--max_per_cell", type=int, default=1,
                        help="Maximum number of points to select per grid cell (default: 1).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate a plot showing the grid selection for each input file.")
    parser.add_argument("--output_suffix", default="_filtered",
                        help="Suffix to append to the original structure filename for output (default: '_filtered').")

    args = parser.parse_args()

    # Expand wildcards in input coordinate files
    expanded_coord_files = []
    for pattern in args.coord_files:
        expanded_coord_files.extend(glob.glob(pattern))

    if not expanded_coord_files:
        print("Error: No coordinate files found matching the input pattern(s).")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Starting Structure Filtering ---")
    print(f"Input Structures: {args.input_structure}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Grid Interval: {args.interval}, Max Points/Cell: {args.max_per_cell}")
    print(f"Processing {len(expanded_coord_files)} coordinate file(s)...")

    total_selected_count = 0

    for coord_filepath in expanded_coord_files:
        print(f"\n--- Processing: {os.path.basename(coord_filepath)} ---")
        try:
            # 1. Load coordinate data
            try:
                data_2d = np.load(coord_filepath)
            except Exception as e:
                print(f"Error loading coordinate file {coord_filepath}: {e}")
                continue # Skip to next file

            print(f"Loaded {data_2d.shape[0]} points from {os.path.basename(coord_filepath)}.")
            if data_2d.shape[0] == 0:
                print("Skipping empty coordinate file.")
                continue

            # 2. Select indices using grid
            selected_indices, x_bins, y_bins = select_indices_from_grid(
                data_2d, args.interval, args.max_per_cell
            )

            total_selected_count += len(selected_indices)

            # 3. Plotting (Optional)
            if args.plot:
                 plot_title = f"Grid Selection ({os.path.basename(coord_filepath)})"
                 plot_grid_selection(data_2d, selected_indices, x_bins, y_bins, title=plot_title)

            # 4. Filter and save structures
            if selected_indices.size > 0:
                # Construct output filename
                base_struct_name = os.path.basename(args.input_structure)
                name_part, ext_part = os.path.splitext(base_struct_name)

                # Include info from coord file name in output name for traceability
                coord_basename = os.path.basename(coord_filepath)
                coord_name_part = coord_basename.replace(".npy", "") # Remove extension

                # Example output name: input_filtered_coords_2d_b_...._tsne_2.data
                # You might want a shorter / more predictable name
                # Let's use a simpler one: {original_struct_name}_{coord_file_info_suffix}.ext
                coord_suffix = coord_name_part.replace("coords_2d","").strip('_') # Extract relevant part
                output_filename = f"{name_part}{args.output_suffix}_{coord_suffix}{ext_part}"
                # Fallback if coord_suffix is too complex
                # output_filename = f"{name_part}{args.output_suffix}_{len(selected_indices)}{ext_part}"

                output_filepath = os.path.join(args.output_dir, output_filename)

                filter_and_save_structures(
                    selected_indices, args.input_structure, output_filepath
                )
            else:
                print("No structures selected based on this coordinate file, skipping structure output.")

        except (ValueError, FileNotFoundError, ImportError, IOError) as e:
            print(f"Error processing {coord_filepath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {coord_filepath}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging unexpected errors

    print(f"\n--- Filtering Complete ---")
    print(f"Total points selected across all files (sum of unique selections per file): {total_selected_count}")


if __name__ == '__main__':
    main()
