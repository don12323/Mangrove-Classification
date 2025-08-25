"""
script for testing different metrics for k-means + my attempt at parallelising Sihloutte score >> UPDATE: it works!

1. Sihlouette score: https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
2. 

"""
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch
from sklearn import metrics


from typing import Tuple, Union
import warnings

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


class GPUSilhouetteScore:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the silhouette score calculator.
        
        Args:
            device: The device to perform calculations on ('cuda' or 'cpu')
        """
        self.device = device
        if device == "cpu":
            warnings.warn("GPU not available, falling back to CPU calculations")

    def _pairwise_distances(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Calculate pairwise distances between two sets of points.
        
        Args:
            X: First set of points (n x d tensor) 
            Y: Second set of points (m x d tensor)
            
        Returns:
            Distances matrix (n x m tensor) 
            Sqrt of l.h.s -> ||x-y||^2 = ||x||^2 + ||y||^2 -2<x,y> (L^2 norm)
        """
        X_norm = (X ** 2).sum(1).view(-1, 1)
        Y_norm = (Y ** 2).sum(1).view(1, -1)
        distances = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
        return torch.clamp(distances, min=0.0).sqrt()

    def _batch_silhouette(self, 
                         data: torch.Tensor, 
                         labels: torch.Tensor, 
                         batch_size: int = 100) -> torch.Tensor:
        """
        Calculate silhouette scores in batches to manage memory.
        
        Args:
            data: Input data tensor
            labels: Cluster labels tensor
            batch_size: Number of points to process at once
            
        Returns:
            Tensor of silhouette scores
        """
        n_samples = data.shape[0]
        silhouette_vals = torch.zeros(n_samples, device=self.device)
        unique_labels = torch.unique(labels)

        for start_idx in tqdm(range(0, n_samples, batch_size), 
                            desc="Calculating silhouette scores"):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_points = data[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            a_i = torch.zeros(batch_points.shape[0], device=self.device)
            
            b_i = torch.full((batch_points.shape[0],), float('inf'), device=self.device)

            for label in unique_labels:
                mask = (labels == label)
                cluster_points = data[mask]
                
                distances = self._pairwise_distances(batch_points, cluster_points)
                
                same_cluster = (batch_labels == label)
                if same_cluster.any():
                    a_i[same_cluster] = (distances[same_cluster].sum(1) - 0) / (mask.sum() - 1)

                different_cluster = (batch_labels != label)
                if different_cluster.any() and mask.any():
                    mean_dist = distances[different_cluster].mean(1)
                    b_i[different_cluster] = torch.minimum(b_i[different_cluster], mean_dist)

            # Calculate silhouette score for batch
            max_val = torch.maximum(a_i, b_i)
            silhouette_vals[start_idx:end_idx] = (b_i - a_i) / max_val

        return silhouette_vals

    def calculate(self, 
                 data: Union[np.ndarray, torch.Tensor], 
                 labels: Union[np.ndarray, torch.Tensor], 
                 batch_size: int = 100) -> Tuple[float, np.ndarray]:
        """
        Calculate silhouette scores for the dataset.
        
        Args:
            data: Input data (numpy array or torch tensor)
            labels: Cluster labels (numpy array or torch tensor)
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (mean silhouette score, individual silhouette scores)
        """
        # Convert input to torch tensors if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        # Move data to appropriate device
        data = data.to(self.device)
        labels = labels.to(self.device)

        # Calculate silhouette scores
        silhouette_vals = self._batch_silhouette(data, labels, batch_size)

        # Convert results back to numpy
        silhouette_vals_np = silhouette_vals.cpu().numpy()
        mean_score = float(silhouette_vals.mean().cpu().numpy())

        return mean_score, silhouette_vals_np

def estimate_memory_usage(n_samples, n_features, batch_size):
    """
    Estimate memory usage for the calculation
    """
    batch_memory = (batch_size * n_samples * 4)/1024**3    # in GB
    print(f"Estimated memory per batch: {batch_memory:.2f} GB")
    print(f"Total number of batches: {(n_samples // batch_size + 1)**2}")
    return batch_memory

if __name__ == "__main__":
    """for testing the metrics on random data instead of running the whole k_means.py code....."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 100000
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
    sh_gpu = GPUSilhouetteScore()
    gpu_score, individual_scores = sh_gpu.calculate(
            data=data,
            labels=labels,
            batch_size=4096
    )
    print(f"Score range: [{individual_scores.min():.4f}, {individual_scores.max():.4f}]")
    print(f"Silhouette Score (GPU): {gpu_score:.3f}")
    cpu_score=metrics.silhouette_score(data, labels, metric='euclidean')
    print(f"Silhoutte Score skleanr (CPU):{cpu_score:.3f}")
    cpump_score = cpu_silhouette_score(data, labels)
    print(f"Silhouette Score (CPUwMP): {cpump_score:.3f}")

