import numpy as np
from typing import List
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def timeseries_clustering_kmeans(
    data: List[np.ndarray],
    n_clusters: int = 3
) -> np.ndarray:
    """
    DTW + KMeansで時系列をクラスタリングする関数。

    Args:
        data (List[np.ndarray]): 各要素が (T,) の1次元配列のリスト。
        n_clusters (int): クラスタ数。

    Returns:
        np.ndarray: shape (N,), 各時系列のクラスタラベル。
    """
    N = len(data)
    if N == 0:
        raise ValueError("Empty data list for KMeans clustering.")
    if n_clusters > N:
        raise ValueError(f"n_clusters={n_clusters} > number of series={N}.")

    dataset = to_time_series_dataset(data)  # shape (N, max_len, 1)
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    labels = km.fit_predict(dataset)
    return labels


def timeseries_clustering_hierarchical(
    data: List[np.ndarray],
    n_clusters: int = 3
) -> np.ndarray:
    """
    DTW距離行列を用いた階層的クラスタリング。

    Args:
        data (List[np.ndarray]): 各要素が (T,) の1次元配列。
        n_clusters (int): クラスタ数。

    Returns:
        np.ndarray: shape (N,), クラスタラベル。
    """
    N = len(data)
    if N == 0:
        raise ValueError("Empty data list for hierarchical clustering.")
    if n_clusters > N:
        raise ValueError(f"n_clusters={n_clusters} > number of series={N}.")

    dist_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            dist = dtw(data[i], data[j])
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels