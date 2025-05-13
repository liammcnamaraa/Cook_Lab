import kmapper as km
from sklearn.cluster import DBSCAN
import TDA
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def kmap_visual():
    cov = pd.read_csv("/Users/douglascook/Cook_Lab/data/joe_data/Pre_0_covariance.csv", index_col=0).to_numpy()
    var = np.diag(cov)
    corr = cov / np.sqrt(np.outer(var, var))
    dist = np.sqrt(2 * (1 - corr))


    mapper = km.KeplerMapper(verbose=1)

    X = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(dist)
    X = StandardScaler().fit_transform(X)
    proj    = mapper.fit_transform(X, projection=[0,1])
    cover   = km.Cover(n_cubes=8, perc_overlap=0.15)
    cluster = DBSCAN(eps=0.4, min_samples=5)
    graph   = mapper.map(proj, X, cover=cover, clusterer=cluster)

    mapper.visualize(graph,
                    path_html="brain_communities.html",
                    title="Brain-Region Communities")
    
if __name__ == "__main__":
    kmap_visual()