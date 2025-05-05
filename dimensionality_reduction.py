# dimensionality_reduction.py
"""
Performs dimensionality reduction on high-dimensional data vectors using
various techniques like t-SNE, PCA, and UMAP.
"""

import numpy as np
import sys
from typing import Optional

# Optional imports - only load if the method is requested
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
try:
    import umap
except ImportError:
    umap = None


def reduce_dimensions(
    data: np.ndarray,
    mode: str = 'tsne',
    n_components: int = 2,
    random_state: Optional[int] = 123415,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: [float, str] = 200.0, # 'auto' is also an option in newer scikit-learn
    tsne_n_iter: int = 1000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1
) -> np.ndarray:
    """
    Reduces the dimensionality of the input data using the specified method.

    Args:
        data (np.ndarray): The high-dimensional data (n_samples, n_features).
        mode (str): The reduction method ('tsne', 'pca', 'umap'). Lowercase.
        n_components (int): The target number of dimensions (usually 2 or 3).
        random_state (Optional[int]): Random seed for stochastic methods (t-SNE, UMAP).
        tsne_perplexity (float): Perplexity parameter for t-SNE.
        tsne_learning_rate (float | str): Learning rate for t-SNE.
        tsne_n_iter (int): Number of iterations for t-SNE.
        umap_n_neighbors (int): Number of neighbors parameter for UMAP.
        umap_min_dist (float): Minimum distance parameter for UMAP.

    Returns:
        np.ndarray: The low-dimensional embedding of the data (n_samples, n_components).

    Raises:
        ValueError: If an unsupported mode is provided or required library is missing.
        ImportError: If the library for the chosen mode is not installed.
    """
    mode = mode.lower()
    print(f"Performing {mode.upper()} dimensionality reduction to {n_components} dimensions...")

    if mode == 'tsne':
        if TSNE is None:
            raise ImportError("scikit-learn is required for t-SNE. Install it: pip install scikit-learn")
        try:
             # Note: perplexity should be less than n_samples
             effective_perplexity = min(tsne_perplexity, data.shape[0] - 1)
             if effective_perplexity != tsne_perplexity:
                  print(f"  Warning: t-SNE perplexity ({tsne_perplexity}) adjusted to {effective_perplexity} (must be < n_samples).")
             if data.shape[0] <= 1:
                  print("  Warning: t-SNE requires more than 1 sample. Returning original data shape potentially.")
                  # Handle cases with 0 or 1 sample if needed, maybe return zeros?
                  return np.zeros((data.shape[0], n_components))


             tsne = TSNE(n_components=n_components,
                         perplexity=effective_perplexity,
                         learning_rate=tsne_learning_rate,
                         n_iter=tsne_n_iter,
                         random_state=random_state,
                         init='pca', # PCA initialization is often faster and more stable
                         n_jobs=-1) # Use all available CPU cores
             data_low_dim = tsne.fit_transform(data)
        except Exception as e:
             print(f"Error during t-SNE: {e}")
             print("Consider adjusting perplexity, learning rate, or checking data consistency.")
             raise # Re-raise the exception

    elif mode == 'pca':
        if PCA is None:
            raise ImportError("scikit-learn is required for PCA. Install it: pip install scikit-learn")
        if data.shape[0] < n_components:
            print(f"  Warning: PCA cannot produce {n_components} components from {data.shape[0]} samples. Adjusting n_components.")
            n_components = data.shape[0]
        if n_components <= 0:
             print("  Warning: Cannot perform PCA with 0 components or samples. Returning zeros.")
             return np.zeros((data.shape[0], n_components if n_components > 0 else 2)) # Default to 2 if 0

        pca = PCA(n_components=n_components, random_state=random_state)
        data_low_dim = pca.fit_transform(data)

    elif mode == 'umap':
        if umap is None:
            raise ImportError("umap-learn is required for UMAP. Install it: pip install umap-learn")
        # Ensure n_neighbors is less than n_samples
        effective_n_neighbors = min(umap_n_neighbors, data.shape[0] - 1)
        if effective_n_neighbors != umap_n_neighbors:
            print(f"  Warning: UMAP n_neighbors ({umap_n_neighbors}) adjusted to {effective_n_neighbors} (must be < n_samples).")
        if effective_n_neighbors <= 1:
             print("  Warning: UMAP requires n_neighbors > 1. Cannot perform reduction.")
             # Return zeros or handle appropriately
             return np.zeros((data.shape[0], n_components))

        try:
            umap_reducer = umap.UMAP(n_components=n_components,
                                 n_neighbors=effective_n_neighbors,
                                 min_dist=umap_min_dist,
                                 random_state=random_state)
            data_low_dim = umap_reducer.fit_transform(data)
        except Exception as e:
             print(f"Error during UMAP: {e}")
             # UMAP can sometimes fail with specific data patterns or parameters
             print("Consider adjusting n_neighbors, min_dist, or checking data.")
             raise

    else:
        raise ValueError(f"Unsupported dimensionality reduction mode: '{mode}'. Choose 'tsne', 'pca', or 'umap'.")

    print(f"{mode.upper()} reduction complete.")
    return data_low_dim