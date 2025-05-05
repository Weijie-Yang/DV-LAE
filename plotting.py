# plotting.py
"""
Handles the creation and display/saving of interactive scatter plots using Plotly
to visualize the low-dimensional embeddings of molecular structures.
"""

import os
import datetime
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import plot as plotly_offline_plot
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

# Optional: Import ASE reader if needed within plotting logic itself
# (Currently handled in main script before calling plot)
# from data_io import read_atomic_structures

def _generate_color_gradient(start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int], num_steps: int) -> List[str]:
    """
    Generates a list of hex color codes interpolating between start and end RGB colors.

    Args:
        start_rgb (Tuple[int, int, int]): RGB tuple for the starting color (0-255).
        end_rgb (Tuple[int, int, int]): RGB tuple for the ending color (0-255).
        num_steps (int): The number of distinct colors to generate.

    Returns:
        List[str]: A list of hex color strings (e.g., '#RRGGBB').
    """
    if num_steps <= 0:
        return []
    if num_steps == 1:
         r, g, b = start_rgb
         return [f'#{r:02X}{g:02X}{b:02X}']

    colors = []
    r1, g1, b1 = start_rgb
    r2, g2, b2 = end_rgb

    for i in range(num_steps):
        # Linear interpolation factor
        factor = i / (num_steps - 1) if num_steps > 1 else 0

        new_r = int(r1 + (r2 - r1) * factor)
        new_g = int(g1 + (g2 - g1) * factor)
        new_b = int(b1 + (b2 - b1) * factor)

        # Clamp values to 0-255 range
        new_r = min(max(new_r, 0), 255)
        new_g = min(max(new_g, 0), 255)
        new_b = min(max(new_b, 0), 255)

        colors.append(f'#{new_r:02X}{new_g:02X}{new_b:02X}')

    return colors


def plot_interactive_scatter(
    data_2d: np.ndarray,
    reduction_mode: str,
    feature_filename: str, # Basename used for saving
    num_interval_bins: int,
    distance_mode: int,
    output_dir: str,
    save_name_override: Optional[str] = None,
    index_dict: Optional[Dict[str, List[int]]] = None, # Grouping by composition
    structure_indices: Optional[List[int]] = None, # Original indices
    highlight_indices: Optional[List[int]] = None # Indices to highlight (e.g., new structures)
    # Add other parameters if needed for hover text, e.g., energies, file indices
    ) -> None:
    """
    Creates an interactive Plotly scatter plot of the 2D data and saves it as HTML.

    Args:
        data_2d (np.ndarray): The 2D data points (n_samples, 2).
        reduction_mode (str): Name of the reduction method used (for title/filename).
        feature_filename (str): Basename of the original feature file (for saving).
        num_interval_bins (int): Number of bins used (for saving).
        distance_mode (int): Distance mode used (for saving).
        output_dir (str): Directory to save the plot and associated data.
        save_name_override (Optional[str]): If provided, use this exact name for the HTML file (without extension).
        index_dict (Optional[Dict[str, List[int]]]): Dictionary mapping composition string
                                                     to list of 0-based indices. Used for coloring.
        structure_indices (Optional[List[int]]): List mapping plot index to original file index.
                                                  Used for hover text.
        highlight_indices (Optional[List[int]]): List of 0-based indices to plot with a
                                                 different marker/color.
    """
    if data_2d.shape[1] != 2:
        raise ValueError("Input data_2d must have exactly two columns (dimensions).")

    n_points = data_2d.shape[0]
    x_coords, y_coords = data_2d[:, 0], data_2d[:, 1]

    fig = go.Figure()

    # Prepare hover text (e.g., original index)
    hover_texts = [f"Point Index: {i}" for i in range(n_points)]
    if structure_indices and len(structure_indices) == n_points:
         hover_texts = [f"Original Index: {orig_idx}<br>Plot Index: {i}"
                        for i, orig_idx in enumerate(structure_indices)]
         # Can add more info here if passed, e.g., energy

    # --- Plotting Logic ---
    if index_dict:
        # Color points by composition group using a color gradient
        # Sort compositions for consistent coloring (optional but good practice)
        sorted_compositions = sorted(index_dict.keys())
        num_groups = len(sorted_compositions)

        # Define gradient start/end colors (adjust as desired)
        start_color_rgb = (237, 237, 214) # Light beige/yellow
        end_color_rgb = (104, 166, 124)   # Muted green
        # Or use a built-in Plotly colorscale:
        # import plotly.express.colors as pcolors
        # colors = pcolors.sample_colorscale('Viridis', num_groups) # Example

        colors = _generate_color_gradient(start_color_rgb, end_color_rgb, num_groups)
        color_map = {comp: colors[i] for i, comp in enumerate(sorted_compositions)}

        print(f"Plotting {n_points} points, colored by {num_groups} composition groups.")

        # Add trace for each composition group
        for i, composition in enumerate(sorted_compositions):
            indices = index_dict[composition]
            if not indices: continue # Skip if a composition group is empty

            group_x = x_coords[indices]
            group_y = y_coords[indices]
            group_hover = [hover_texts[idx] for idx in indices]

            fig.add_trace(go.Scatter(
                x=group_x,
                y=group_y,
                mode='markers',
                marker=dict(
                    color=color_map[composition],
                    size=8,
                    opacity=0.8,
                    # symbol='circle' # Default
                    line=dict(width=0.5, color='DarkSlateGrey') # Optional border
                ),
                name=f'Comp: {composition} ({len(indices)})', # Legend entry
                text=group_hover, # Text appearing on hover
                hoverinfo='text+name'
            ))

        # Example of highlighting specific points (e.g., 'new' structures)
        # This part needs refinement based on how `highlight_indices` is defined and used
        if highlight_indices:
             print(f"Highlighting {len(highlight_indices)} points.")
             # Ensure highlight_indices are valid
             valid_highlights = [idx for idx in highlight_indices if 0 <= idx < n_points]
             if not valid_highlights:
                 print("  No valid indices found for highlighting.")
             else:
                 highlight_x = x_coords[valid_highlights]
                 highlight_y = y_coords[valid_highlights]
                 highlight_hover = [hover_texts[idx] for idx in valid_highlights]

                 fig.add_trace(go.Scatter(
                     x=highlight_x,
                     y=highlight_y,
                     mode='markers',
                     marker=dict(
                         color='red', # Distinct highlight color
                         size=10,
                         symbol='star', # Different symbol
                         opacity=1.0,
                         line=dict(width=1, color='black')
                     ),
                     name='Highlighted Points',
                     text=highlight_hover,
                     hoverinfo='text+name'
                 ))

    else:
        # No grouping, plot all points with a single color
        print(f"Plotting {n_points} points with default coloring.")
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                color='blue', # Default color
                size=8,
                opacity=0.7
            ),
            name='All Structures',
            text=hover_texts,
            hoverinfo='text'
        ))

    # --- Layout and Saving ---
    plot_title = f'{reduction_mode.upper()} Visualization ({os.path.basename(feature_filename)})'
    fig.update_layout(
        title=plot_title,
        xaxis_title=f'{reduction_mode.upper()} Dimension 1',
        yaxis_title=f'{reduction_mode.upper()} Dimension 2',
        plot_bgcolor='white',  # White background for plot area
        paper_bgcolor='white', # White background for entire figure
        legend_title='Groups' if index_dict else None,
        hovermode='closest', # Show hover for nearest point
        xaxis=dict(showgrid=False, zeroline=False), # Cleaner axes
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=40, r=40, t=60, b=40) # Adjust margins
    )

    # Determine output filename
    if save_name_override:
        # Use provided name, ensure no extension is included by user
        base_save_name = os.path.splitext(save_name_override)[0]
        html_filename = f"{base_save_name}.html"
    else:
        # Generate automatic filename
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        # Ensure feature_filename doesn't have extension already
        base_feature_name = os.path.splitext(feature_filename)[0]
        html_filename = f'{timestamp}_{base_feature_name}_{num_interval_bins}_{reduction_mode}_{distance_mode}.html'

    output_filepath = os.path.join(output_dir, html_filename)

    # Save the plot as an HTML file
    try:
        # Using pio.write_html is generally preferred for saving
        pio.write_html(fig, output_filepath, auto_open=False)
        print(f"Interactive plot saved to: {output_filepath}")

        # Option to automatically open the plot in a web browser
        # import webbrowser
        # webbrowser.open(f'file://{os.path.realpath(output_filepath)}')

    except Exception as e:
        print(f"Error saving Plotly HTML file: {e}")

    # Optional: Also save using plotly_offline_plot for different behavior/output
    # try:
    #     plotly_offline_plot(fig, filename=output_filepath, auto_open=True)
    #     print(f"Interactive plot saved and opened: {output_filepath}")
    # except Exception as e:
    #     print(f"Error saving/opening Plotly offline file: {e}")