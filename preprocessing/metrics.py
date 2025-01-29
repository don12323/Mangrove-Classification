"""
script for testing different metrics for k-means.

1. Sihlouette score: https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
2.

"""
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch
from sklearn import metrics
    
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

def cpu_silhouette_score(data, labels, n_jobs=None):
    """
    Calculate silhouette score w multiprocessing (too much memory overhead I think 
	when testing for larger samples)
    
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


import numpy as np
import torch
from tqdm import tqdm

def pairwise_distances_batch(x, y, batch_size=1024):
    """
    Compute pairwise distances between two sets of points using double batching
    
    This function processes both dimensions in batches to keep memory usage low.
    For n points, instead of creating an n×n matrix, we create batch_size×batch_size
    matrices and process them sequentially.
    """
    n_samples_x = x.size(0)
    n_samples_y = y.size(0)
    distances = torch.zeros(n_samples_x, device=x.device)
    
    # Process data in batches for both dimensions
    for start_x in range(0, n_samples_x, batch_size):
        end_x = min(start_x + batch_size, n_samples_x)
        batch_x = x[start_x:end_x]
        
        # Calculate batch norm
        batch_x_norm = (batch_x**2).sum(1).view(-1, 1)
        
        for start_y in range(0, n_samples_y, batch_size):
            end_y = min(start_y + batch_size, n_samples_y)
            batch_y = y[start_y:end_y]
            
            # Calculate batch distances
            batch_y_norm = (batch_y**2).sum(1).view(1, -1)
            batch_dist = batch_x_norm + batch_y_norm - 2 * torch.mm(batch_x, batch_y.t())
            
            # Accumulate distances
            batch_dist = torch.clamp(batch_dist, min=0).sqrt()
            distances[start_x:end_x] += batch_dist.sum(dim=1)
            
            # Clear memory
            del batch_dist
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return distances

def gpu_silhouette_score(data, labels, device=None, batch_size=1024):
    """
    Calculate silhouette score using GPU acceleration with memory-efficient batching
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to torch tensors
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Move to GPU
    data = data.to(device)
    labels = labels.to(device)
    
    unique_labels = torch.unique(labels)
    n_samples = len(data)
    
    # Process in batches
    silhouette_values = torch.zeros(n_samples, device=device)
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Calculating silhouette scores"):
        end = min(start + batch_size, n_samples)
        batch_data = data[start:end]
        batch_labels = labels[start:end]
        
        # Calculate a_i for points in same cluster
        for label in unique_labels:
            mask_same = (labels == label)
            if mask_same.sum() > 1:  # More than just the point itself
                same_cluster_points = data[mask_same]
                batch_distances = pairwise_distances_batch(
                    batch_data, 
                    same_cluster_points,
                    batch_size=batch_size
                )
                
                # Update a_i for points in this cluster
                batch_mask = (batch_labels == label)
                if batch_mask.any():
                    silhouette_values[start:end][batch_mask] = batch_distances[batch_mask] / (mask_same.sum() - 1)
        
        # Calculate b_i for points in other clusters
        for label in unique_labels:
            mask_other = (labels == label)
            other_cluster_points = data[mask_other]
            
            batch_distances = pairwise_distances_batch(
                batch_data,
                other_cluster_points,
                batch_size=batch_size
            )
            
            # Update b_i for points not in this cluster
            batch_mask = (batch_labels != label)
            if batch_mask.any():
                mean_dist = batch_distances / mask_other.sum()
                current_b = silhouette_values[start:end]
                silhouette_values[start:end][batch_mask] = torch.minimum(
                    current_b[batch_mask],
                    mean_dist[batch_mask]
                )
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate final silhouette scores
    valid_mask = silhouette_values != 0
    if valid_mask.any():
        silhouette_values[valid_mask] = (silhouette_values[valid_mask] - 1) / torch.maximum(
            silhouette_values[valid_mask],
            torch.ones_like(silhouette_values[valid_mask])
        )
    
    return silhouette_values.mean().cpu().item()

def estimate_memory_usage(n_samples, n_features, batch_size):
    """
    Estimate memory usage for the calculation
    """
    bytes_per_float = 4  # 32-bit float
    batch_memory = (batch_size * batch_size * bytes_per_float) / (1024**3)  # in GB
    print(f"Estimated memory per batch: {batch_memory:.2f} GB")
    print(f"Total number of batches: {(n_samples // batch_size + 1)**2}")
    return batch_memory

if __name__ == "__main__":
    """for testing the metrics on random data instead of running the whole k_means.py code....."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    n_features = 4
    batch_size = 4096
    
    # Estimate memory usage first
    estimate_memory_usage(n_samples, n_features, batch_size=batch_size)
    
    # Generate data
    data = np.concatenate([
        np.random.normal(0, 1, (n_samples, n_features)),
        np.random.normal(4, 1, (n_samples, n_features))
    ])
    labels = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    
    # Calculate score using GPU
    print(">>> Calculating gpu sh score ...")
    gpu_score = gpu_silhouette_score(data, labels, batch_size=4096)
    print(f"Silhouette Score (GPU): {gpu_score:.3f}")
    cpu_score=metrics.silhouette_score(data, labels, metric='euclidean')
    print(f"Silhoutte Score skleanr (CPU):{cpu_score:.3f}")
    cpump_score = cpu_silhouette_score(data, labels)
    print(f"Silhouette Score (CPUwMP): {cpump_score:.3f}")

