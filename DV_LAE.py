# main.py
"""
Main script to perform dimensionality reduction analysis on molecular feature data.

Workflow:
1. Define input parameters (file paths, analysis settings).
2. Read feature data ranges (min/max) from the dataset.
3. Read the reference structure's feature data.
4. Calculate feature histograms for the reference structure.
5. Calculate distance vectors for all structures relative to the reference.
6. Perform dimensionality reduction (t-SNE, PCA, or UMAP) on the distance vectors.
7. (Optional) Read corresponding atomic structures (e.g., from input.data) for plotting context.
8. Generate and save an interactive Plotly scatter plot of the low-dimensional data.
"""

import os
import time
import numpy as np
import datetime
from collections import defaultdict
import argparse  # For command-line argument parsing
from typing import Optional

# Optional imports - only load if the method is requested
# Import functions from our refactored modules
try:
    from molecule_structure import Molecule
    from data_io import (read_single_molecule_features,
                         read_atomic_structures, read_n2p2_output)
    from feature_processing import (calculate_feature_histograms, get_feature_ranges,
                                    compute_all_distance_vectors)
    from dimensionality_reduction import reduce_dimensions
    from plotting import plot_interactive_scatter
    from utils import find_latest_file  # If needed for resuming/checking
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print(
        "Ensure all .py files (molecule_structure.py, data_io.py, etc.) are in the same directory or accessible in PYTHONPATH.")
    exit(1)


def run_analysis_and_plot(
        feature_path: str,
        ref_feature_path: Optional[str] = None,
        structure_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        ref_index: int = 1,
        num_interval_bins: int = 10,
        reduction_mode: str = 'tsne',
        distance_mode: int = 0,
        highlight_last_n: Optional[int] = None,  # Replaces `num` logic
        save_plot_name: Optional[str] = None
) -> None:
    """
    Orchestrates the full analysis workflow.

    Args:
        feature_path (str): Path to the feature data file (e.g., function.data).
        ref_feature_path (Optional[str]): Path to the reference feature file. If None, uses feature_path.
        structure_path (Optional[str]): Path to the corresponding atomic structure file
                                        (e.g., input.data, trajectory.xyz) for grouping/coloring.
        output_dir (Optional[str]): Directory to save outputs (plots, data). If None, uses
                                    the directory of feature_path.
        ref_index (int): 1-based index of the reference structure in ref_feature_path.
                         Negative index counts from the end. 0 is invalid here.
        num_interval_bins (int): Number of bins for histogram features.
        reduction_mode (str): Dimensionality reduction method ('tsne', 'pca', 'umap').
        distance_mode (int): Method for calculating distance vectors (0, 1, or 2).
        highlight_last_n (Optional[int]): If set, highlight the last N points in the plot.
        save_plot_name (Optional[str]): Specific name for the output HTML plot file (without extension).
    """
    start_time = time.time()

    # --- Parameter Validation and Setup ---
    if ref_index == 0:
        raise ValueError("ref_index=0 is invalid. Use 1 for the first structure or negative indices for end.")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    ref_feature_path = ref_feature_path or feature_path  # Default reference path
    if not os.path.exists(ref_feature_path):
        raise FileNotFoundError(f"Reference feature file not found: {ref_feature_path}")

    if output_dir is None:
        output_dir = os.path.dirname(feature_path)
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    reduction_mode = reduction_mode.lower()
    if reduction_mode not in ['tsne', 'pca', 'umap']:
        raise ValueError("reduction_mode must be 'tsne', 'pca', or 'umap'")
    if distance_mode not in [0, 1, 2]:
        raise ValueError("distance_mode must be 0, 1, or 2")
    if num_interval_bins <= 0:
        raise ValueError("num_interval_bins must be positive.")

    print("--- Starting Analysis ---")
    print(f"Feature Data: {feature_path}")
    print(f"Reference Data: {ref_feature_path} (Index: {ref_index})")
    if structure_path: print(f"Atomic Structures: {structure_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Settings: Bins={num_interval_bins}, Reduction={reduction_mode.upper()}, DistMode={distance_mode}")

    # --- 1. Get Feature Ranges and Total Count ---
    print("\nStep 1: Calculating feature value ranges...")
    try:
        max_vals, min_vals, element_list = get_feature_ranges(feature_path)
        if not element_list:
            print("Error: No elements found or features read. Cannot proceed.")
            return
        # Calculate total number of structures (can be slow for large files)
        total_structures = read_single_molecule_features(feature_path, index=0)
        print(f"Found {total_structures} structures and elements: {element_list}")
    except Exception as e:
        print(f"Error during feature range calculation or counting: {e}")
        return

    # --- 2. Calculate Intervals and Read Reference Molecule ---
    print(f"\nStep 2: Reading reference structure (Index {ref_index})...")
    intervals = {}
    for elem in element_list:
        # Avoid division by zero if min==max
        range_vals = max_vals[elem] - min_vals[elem]
        intervals[elem] = np.divide(range_vals, num_interval_bins,
                                    out=np.zeros_like(range_vals),  # Output 0 where division by zero occurs
                                    where=(range_vals > 1e-9))  # Small threshold for float comparison

    try:
        if ref_index < 0:
            ref_molecule, _ = read_single_molecule_features(ref_feature_path, index=ref_index)
            print(f"  (Read {abs(ref_index)} from end, actual index {ref_molecule.index})")
        else:
            ref_molecule = read_single_molecule_features(ref_feature_path, index=ref_index)
    except (IndexError, ValueError, FileNotFoundError) as e:
        print(f"Error reading reference structure: {e}")
        return

    # --- 3. Calculate Histograms for Reference ---
    print("\nStep 3: Calculating feature histograms for reference structure...")
    ref_hist_data = calculate_feature_histograms(
        ref_molecule, max_vals, min_vals, intervals, element_list, num_interval_bins
    )

    # --- 4. Compute Distance Vectors for All Structures ---
    print("\nStep 4: Computing distance vectors relative to reference...")
    distance_vectors = compute_all_distance_vectors(
        feature_path, ref_hist_data, max_vals, min_vals, intervals,
        element_list, total_structures, num_interval_bins, distance_mode
    )

    if not distance_vectors:
        print("Error: Failed to compute any distance vectors.")
        return
    distance_matrix = np.array(distance_vectors)  # Convert list of arrays to a 2D numpy matrix

    # --- 5. Perform Dimensionality Reduction ---
    print("\nStep 5: Performing dimensionality reduction...")
    try:
        data_2d = reduce_dimensions(distance_matrix, mode=reduction_mode)
        # Basic check on output shape
        if data_2d.shape[0] != total_structures or data_2d.shape[1] != 2:
            print(
                f"Warning: Dimensionality reduction output shape unexpected ({data_2d.shape}). Expected ({total_structures}, 2).")
            # Decide how to proceed, maybe pad or raise error? Continuing for now.
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        return

    # --- 6. (Optional) Prepare Grouping/Highlighting Info ---
    print("\nStep 6: Preparing plot data (grouping/highlighting)...")
    index_dict_for_plot = None
    structure_indices_map = list(range(total_structures))  # Default mapping: plot index = file index
    highlight_indices_list = None
    atoms_list = None  # Store Atoms objects if read

    if structure_path:
        print(f"  Reading atomic structures from {structure_path} for grouping...")
        try:
            # Try n2p2 first to get line numbers if needed, fallback to generic ASE
            file_ext = os.path.splitext(structure_path)[1].lower()
            line_indices = None
            if file_ext == '.data':
                try:
                    # Read all, get original line indices too
                    atoms_list, line_indices = read_n2p2_output(filename=structure_path, index=':')
                    print(f"  Read {len(atoms_list)} structures using n2p2 reader.")
                except Exception as e:
                    print(f"  n2p2 read failed ({e}), falling back to generic ASE reader.")
                    atoms_list = None  # Reset atoms_list

            if atoms_list is None:  # If not read by n2p2 or different extension
                atoms_list = read_atomic_structures(structure_path, index=':')
                print(f"  Read {len(atoms_list)} structures using generic ASE reader.")

            if len(atoms_list) != total_structures:
                print(f"  Warning: Number of structures in {structure_path} ({len(atoms_list)}) "
                      f"does not match feature file ({total_structures}). Grouping might be incorrect.")
                # Decide how to handle: disable grouping, try to align? Disabling for safety.
                atoms_list = None
            else:
                # Generate index dictionary based on composition (sorted symbols string)
                index_dict_for_plot = defaultdict(list)
                for i, atoms in enumerate(atoms_list):
                    comp_str = atoms.get_chemical_formula(mode='hill', empirical=False)  # Or use sorted symbols
                    index_dict_for_plot[comp_str].append(i)
                print(f"  Grouped structures into {len(index_dict_for_plot)} composition groups.")
                # If line_indices were read, could potentially map plot index to original line index
                # structure_indices_map = line_indices # This assumes a 1-to-1 mapping holds

        except Exception as e:
            print(f"  Error reading or processing structure file {structure_path}: {e}. Plot will not be grouped.")
            index_dict_for_plot = None

    # Define indices to highlight (e.g., last N points if requested)
    if highlight_last_n is not None and highlight_last_n > 0:
        if highlight_last_n >= total_structures:
            print("  Highlighting all points.")
            highlight_indices_list = list(range(total_structures))
        else:
            highlight_indices_list = list(range(total_structures - highlight_last_n, total_structures))
            print(f"  Highlighting the last {highlight_last_n} points.")
    elif highlight_last_n is not None and highlight_last_n <= 0:
        print("  No points highlighted (highlight_last_n <= 0).")

    # --- 7. Generate and Save Plot ---
    print("\nStep 7: Generating interactive plot...")
    try:
        feature_file_basename = os.path.basename(feature_path)
        plot_interactive_scatter(
            data_2d=data_2d,
            reduction_mode=reduction_mode,
            feature_filename=feature_file_basename,
            num_interval_bins=num_interval_bins,
            distance_mode=distance_mode,
            output_dir=output_dir,
            save_name_override=save_plot_name,
            index_dict=index_dict_for_plot,
            structure_indices=structure_indices_map,  # Pass mapping for hover text
            highlight_indices=highlight_indices_list
        )
    except Exception as e:
        print(f"Error generating or saving the plot: {e}")
        # Optionally save intermediate data here if plotting fails

    # --- 8. Save Intermediate Data (Optional) ---
    print("\nStep 8: Saving intermediate data (distance vectors, 2D coordinates)...")
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        base_feature_name = os.path.splitext(os.path.basename(feature_path))[0]
        dist_vec_filename = f'{timestamp}_{base_feature_name}_{num_interval_bins}_{reduction_mode}_{distance_mode}_dist_vectors.npy'
        coords_filename = f'{timestamp}_{base_feature_name}_{num_interval_bins}_{reduction_mode}_{distance_mode}_coords_2d.npy'
        coords_csv_filename = f'{timestamp}_{base_feature_name}_{num_interval_bins}_{reduction_mode}_{distance_mode}_coords_2d.csv'

        np.save(os.path.join(output_dir, dist_vec_filename), distance_matrix)
        np.save(os.path.join(output_dir, coords_filename), data_2d)
        # Save 2D coords also as CSV for easier external use
        np.savetxt(os.path.join(output_dir, coords_csv_filename), data_2d, delimiter=',', fmt='%.6f',
                   header='Dim1,Dim2', comments='')

        print(f"  Distance vectors saved to: {dist_vec_filename}")
        print(f"  2D coordinates saved to: {coords_filename} / .csv")

        # Save grouping info if generated
        if index_dict_for_plot:
            grouping_filename = f'{timestamp}_{base_feature_name}_{num_interval_bins}_{reduction_mode}_{distance_mode}_grouping.npy'
            # Need to save dict properly, np.save works directly on dicts
            np.save(os.path.join(output_dir, grouping_filename), index_dict_for_plot)
            print(f"  Grouping dictionary saved to: {grouping_filename}")

    except Exception as e:
        print(f"  Warning: Failed to save intermediate data: {e}")

    end_time = time.time()
    print("\n--- Analysis Complete ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Default Settings (as in the original script) ---
    default_feature_path = r'function.data'
    default_ref_feature_path = None  # Uses feature_path by default
    default_structure_path = r'input.data'  # Path to atomic structures (optional)
    default_output_dir = None  # Saves in the same dir as feature_path by default

    default_ref_index = 1  # Compare to the first structure (1-based)
    default_num_bins = 20  # Number of histogram bins (was hardcoded loop variable before)
    default_reduction_mode = 'tsne'  # 'tsne', 'pca', or 'umap'
    default_distance_modes = [2]  # Distance calculation mode(s) (was hardcoded list before)
    default_highlight_last = None  # Optional: number of final points to highlight (e.g., 5)

    # --- Argument Parser Setup (Optional but recommended) ---
    parser = argparse.ArgumentParser(description="Dimensionality Reduction Analysis for Molecular Features")
    parser.add_argument('--feature', type=str, default=default_feature_path,
                        help=f"Path to the feature file (default: {default_feature_path})")
    parser.add_argument('--ref_feature', type=str, default=default_ref_feature_path,
                        help="Path to the reference feature file (default: same as --feature)")
    parser.add_argument('--structure', type=str, default=default_structure_path,
                        help=f"Path to atomic structure file for grouping (optional, default: {default_structure_path})")
    parser.add_argument('--outdir', type=str, default=default_output_dir,
                        help="Directory for output files (default: same as feature file)")
    parser.add_argument('--ref_idx', type=int, default=default_ref_index,
                        help=f"1-based index of reference structure (neg counts from end, default: {default_ref_index})")
    parser.add_argument('--bins', type=int, default=default_num_bins,
                        help=f"Number of histogram bins (default: {default_num_bins})")
    parser.add_argument('--mode', type=str, default=default_reduction_mode, choices=['tsne', 'pca', 'umap'],
                        help=f"Dimensionality reduction mode (default: {default_reduction_mode})")
    parser.add_argument('--dist_mode', type=int, nargs='+', default=default_distance_modes, choices=[0, 1, 2],
                        help=f"Distance vector mode(s) (default: {default_distance_modes})")
    parser.add_argument('--highlight', type=int, default=default_highlight_last,
                        help="Highlight the last N points in the plot (optional)")
    parser.add_argument('--savename', type=str, default=None,
                        help="Override automatic plot filename (provide name without extension)")

    args = parser.parse_args()

    # --- Execute Workflow for each distance mode ---
    overall_start_time = time.time()

    # Use lists for bins and dist_modes if you want to iterate through combinations
    num_bins_list = [args.bins]  # Or loop: [10, 20, 30]
    distance_modes_list = args.dist_mode  # Already a list from nargs='+'

    for num_bins_val in num_bins_list:
        for dist_mode_val in distance_modes_list:
            print(f"\n=== Running analysis for Bins={num_bins_val}, DistMode={dist_mode_val} ===")
            try:
                run_analysis_and_plot(
                    feature_path=args.feature,
                    ref_feature_path=args.ref_feature,
                    structure_path=args.structure if os.path.exists(args.structure or '') else None,
                    # Pass None if file doesn't exist
                    output_dir=args.outdir,
                    ref_index=args.ref_idx,
                    num_interval_bins=num_bins_val,
                    reduction_mode=args.mode,
                    distance_mode=dist_mode_val,
                    highlight_last_n=args.highlight,
                    save_plot_name=args.savename
                )
            except (FileNotFoundError, ValueError, IndexError, ImportError) as e:
                print(f"\n*** Error encountered during analysis run: {e} ***")
                print("*** Skipping this combination and continuing if possible... ***")
            except Exception as e:  # Catch other unexpected errors
                print(f"\n*** An unexpected error occurred: {e} ***")
                import traceback

                traceback.print_exc()  # Print detailed traceback for debugging
                print("*** Skipping this combination and continuing if possible... ***")

    overall_end_time = time.time()
    print(f"\n=== All analyses finished. Total time: {overall_end_time - overall_start_time:.2f} seconds ===")