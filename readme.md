## What is DV-LAE ?

DV-LAE (Diversity Visualizer of Local Atomic Environments) is a toolkit developed based on Python language, which aims to realize statistics, screening and visualization of diversity through symmetric function analysis and screening of potential function training data. This toolkit can help users analyze the diversity of sample data while effectively saving training costs.
## Features

*   Reads feature data from `function.data`-like files.
*   Reads atomic structure data from n2p2 `output.data` format or any format supported by ASE (e.g., XYZ, POSCAR) for optional coloring/grouping.
*   Calculates feature histograms and distance vectors between structures using different modes:
    *   Mode 0: Binary difference per feature (1 if ranges differ significantly, 0 otherwise).
    *   Mode 1: Element-wise absolute difference between histogram counts.
    *   Mode 2: Binary difference per histogram bin (1 if one bin has counts and the other doesn't, 0 otherwise).
*   Performs dimensionality reduction using:
    *   t-SNE (t-Distributed Stochastic Neighbor Embedding)
    *   PCA (Principal Component Analysis)
    *   UMAP (Uniform Manifold Approximation and Projection)
*   Generates interactive 2D scatter plots using Plotly.
*   Optionally colors points by chemical composition based on provided atomic structure files.
*   Optionally highlights the last N points in the dataset.
*   Saves plots as standalone HTML files.
*   Saves intermediate data (distance vectors, 2D coordinates) as NumPy (`.npy`) and CSV (`.csv`) files.
*   Configurable via command-line arguments.
<iframe width="560" height="315" src="https://www.youtube.com/embed/J8WpREFhcHY" title="DV-LAE Methodology Implementation Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## File Structure
```
├── DV_LAE.py # Main execution script, orchestrates the workflow
├── molecule_structure.py # Defines the Molecule class
├── data_io.py # Functions for reading feature and structure files
├── feature_processing.py # Functions for histogramming and distance calculation
├── dimensionality_reduction.py # Functions for t-SNE, PCA, UMAP
├── plotting.py # Functions for creating Plotly plots
├── utils.py # Utility functions (e.g., finding files, deleting structures)
└── README.md # This file
```


## Requirements

*   Python 3.7+ recommended
*   Libraries:
    *   NumPy
    *   ASE (Atomic Simulation Environment)
    *   tqdm (for progress bars)
    *   Plotly (for interactive plotting)
    *   scikit-learn (for t-SNE, PCA)
    *   umap-learn (for UMAP)

You can install the required libraries using pip:

```
pip install numpy ase tqdm plotly scikit-learn "umap-learn>=0.5"
```

## Usage
The main script to run the analysis is DV_LAE.py.

## Basic Execution
To run the analysis with default settings (using function.data, input.data in the current directory, comparing to the 1st structure, using t-SNE, 20 bins, and distance mode 2):
```
python DV_LAE.py
```
Command-Line Arguments
You can customize the behavior using command-line arguments. Run python DV_LAE.py --help to see all options:
```
usage: DV_LAE.py [-h] [--feature FEATURE] [--ref_feature REF_FEATURE] [--structure STRUCTURE] [--outdir OUTDIR] [--ref_idx REF_IDX] [--bins BINS] [--mode {tsne,pca,umap}]
               [--dist_mode {0,1,2} [{0,1,2} ...]] [--highlight HIGHLIGHT] [--savename SAVENAME]

Dimensionality Reduction Analysis for Molecular Features

options:
  -h, --help            show this help message and exit
  --feature FEATURE     Path to the feature file (default: function.data)
  --ref_feature REF_FEATURE
                        Path to the reference feature file (default: same as --feature)
  --structure STRUCTURE
                        Path to atomic structure file for grouping (optional, default: input.data)
  --outdir OUTDIR       Directory for output files (default: same as feature file)
  --ref_idx REF_IDX     1-based index of reference structure (neg counts from end, default: 1)
  --bins BINS           Number of histogram bins (default: 20)
  --mode {tsne,pca,umap}
                        Dimensionality reduction mode (default: tsne)
  --dist_mode {0,1,2} [{0,1,2} ...]
                        Distance vector mode(s) (default: [2])
  --highlight HIGHLIGHT
                        Highlight the last N points in the plot (optional)
  --savename SAVENAME   Override automatic plot filename (provide name without extension)
```

## Examples
### 1. Run PCA using 15 bins and distance mode 1:
~~~b
python DV_LAE.py --mode pca --bins 15 --dist_mode 1
~~~

### 2. Compare against the 10th structure from the end (-10), use UMAP, specify output directory:

~~~
python DV_LAE.py --ref_idx -10 --mode umap --outdir ./umap_results
~~~

### 3. Run using different feature and structure files, highlight last 5 structures:

~~~
python DV_LAE.py --feature my_features.dat --structure trajectory.xyz --highlight 5
~~~

### 4. Run analysis for multiple distance modes (e.g., 0 and 2) sequentially:

~~~
python DV_LAE.py --dist_mode 0 2 --bins 25
~~~
(The script will run the full analysis first for mode 0, then again for mode 2)

### 5. Run without using atomic structure data for coloring:

~~~
python DV_LAE.py --structure ""
~~~
(Or ensure the default input.data does not exist)

## Input Data Format
* Feature File (--feature, e.g., function.data):
    *  Expected to contain blocks of data, each representing one structure.
    *  Each block starts with a line containing the number of atoms (N).
    *  The next N lines contain the feature vector for each atom: atomic_number G1 G2 G3 ...
    *  Blocks are typically separated by a blank line (the code handles this).
    *  Comment lines starting with # are ignored.
* Structure File (--structure, e.g., input.data or trajectory.xyz):
    *  If the file extension is .data, it's assumed to be in n2p2 output format (begin/end blocks, lattice, atom ...).
    *  For other extensions (e.g., .xyz, .cif, POSCAR), the file is read using ASE's generic ase.io.read.
    *  This file is optional. If provided and successfully read, it's used to group points in the plot by chemical composition. The number of structures must match the feature file.

## Output
The script generates the following files in the specified output directory (--outdir or the directory of the feature file):
1. Interactive Plot (.html): An HTML file containing the Plotly scatter plot. The filename includes a timestamp and the analysis parameters (e.g., 202310271130_function_20_tsne_2.html). Can be overridden with --savename.
2. Distance Vectors (.npy): A NumPy binary file containing the raw, high-dimensional distance vectors for all structures relative to the reference. (e.g., ..._dist_vectors.npy)
3. 2D Coordinates (.npy): A NumPy binary file containing the calculated 2D coordinates after dimensionality reduction. (e.g., ..._coords_2d.npy)
4. 2D Coordinates (.csv): A CSV text file containing the 2D coordinates (Dim1, Dim2). (e.g., ..._coords_2d.csv)
5. Grouping Info (.npy, optional): If structures were grouped by composition, a NumPy file saving the dictionary that maps composition strings to structure indices. (e.g., ..._grouping.npy)

