import sys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def dimensionality_reduction(jiangweilist, mode):
    if mode.lower() == 'tsne':
        tsne = TSNE(n_components=2, perplexity=50, learning_rate=10, n_iter=1000)
        data_2d = tsne.fit_transform(np.array(jiangweilist))
    elif mode.lower() == 'pca':
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(jiangweilist)
    else:
        sys.exit(-2)

    return data_2d
