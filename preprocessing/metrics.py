"""
script for testing different metrics for k-means.

1. Sihlouette score: https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
2.

"""
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
    
def process_point(args):
    point, point_idx, data, labels, unique_labels = args
    cluster = labels[point_idx]
    
    # Calc a_i
    same_cluster = data[labels == cluster]
    a_i = np.mean(cdist([point], same_cluster)[0])
    # Calc b_i
    b_i = np.inf
    for other_cluster in unique_labels:
        if other_cluster != cluster:
            other_cluster_points = data[labels == other_cluster]
            mean_dist = np.mean(cdist([point], other_cluster_points)[0])
            b_i = min(b_i, mean_dist)
    
    s_i = (b_i - a_i) / max(a_i, b_i)
    return s_i

def calc_silhouette_score(data, labels, n_jobs=None):
    """
    Calculate silhouette score w multiprocessing
    
    Params:
    -----------
    data : 2D data array to analyse (n_samples, n_features).
    
    labels : Cluster labels for each point.

    n_jobs : int, optional. Number of processes to use. 
            Defaults to number of CPU cores minus 1
        
    Returns:
    --------
    float
        Mean silhouette score for the dataset
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    unique_labels = np.unique(labels)
    
    # Prepare args for mp
    args = [(point, idx, data, labels, unique_labels) 
            for idx, point in enumerate(data)]
    
    # Calc scores in parallel (using pool.imap here to load results one at a time which is mem efficient)
    with Pool(processes=n_jobs) as pool:
        silhouette_values = list(tqdm(
            pool.imap(process_point, args),
            total=len(data),
            desc="Calculating silhouette scores"
        ))
    
    return np.mean(silhouette_values)


#if __main__ == '__main__':

    """for testing the metrics on random data instead of running the whole k_means.py code....."""
