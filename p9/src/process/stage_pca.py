import numpy as np
import pandas as pd
import scipy.spatial.distance as sp_distance
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def split_data(df):
    """
    Split the input dataframe into a dictionary containing data matrix, samples, and features.

    Parameters:
    - df (DataFrame): Input dataframe.

    Returns:
    dict: A dictionary with keys 'matrix', 'samples', and 'features'.
    """ 
    # Split the dataframe: data matrix, samples, features
    matrix = df.values
    samples = {f"S{i}": v for i, v in enumerate(list(df.index))}
    features = {f"F{i}": v for i, v in enumerate(list(df.columns))}
    dataset = {'matrix': matrix, 'scaled_matrix': scale_matrix(matrix), 'samples': samples, 'features': features, 'dataframe': df}
    return dataset

def scale_matrix(matrix):
    """
    Scale the input matrix using QuantileTransformer.

    Parameters:
    - matrix (numpy.ndarray): Input matrix.

    Returns:
    numpy.ndarray: Scaled matrix.
    """
    # Input validation
    if len(matrix.shape) != 2:
        raise ValueError("Input matrix must be a 2D array.")
    # Scaling using QuantileTransformer
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    scaled_matrix = scaler.fit_transform(matrix)
    # Check mean and std (commented out for now)
    # if not np.allclose(scaled_matrix.mean(), 0) or not np.allclose(scaled_matrix.std(), 1):
    #     raise Exception(f"Erreur, moyenne = {scaled_matrix.mean()}\nSTD = {scaled_matrix.std()}")
    return scaled_matrix

def perform_pca(scaled_matrix, num_components):
    """
    Perform Principal Component Analysis (PCA) on the input matrix.

    Parameters:
    - scaled_matrix (numpy.ndarray): Scaled input matrix.
    - num_components (int): Number of principal components to retain.

    Returns:
    dict: A dictionary containing PCA results.
    """
    # PCA using sklearn
    pca = PCA(n_components=num_components)
    fitted_pca_model = pca.fit(scaled_matrix)
    transformed_matrix = pca.transform(scaled_matrix)

    pca_results = {
        'fitted_pca_model': fitted_pca_model,
        'transformed_matrix': transformed_matrix,
        'explained_variance_ratio': (fitted_pca_model.explained_variance_ratio_ * 100).round(),
        'principal_components': fitted_pca_model.components_
    }
    return pca_results

def perform_kmeans(matrix, n_clusters):
    """
    Perform k-means clustering on the input matrix.

    Parameters:
    - matrix (numpy.ndarray): Input matrix.
    - n_clusters (int): Number of clusters.

    Returns:
    tuple: A tuple containing k-means model and cluster map.
    """
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100)
    kmeans.fit(matrix)

    # Mapper
    cluster_map = {str(k): [] for k in range(n_clusters)}
    for i, v in enumerate(kmeans.labels_):
        cluster_map[str(v)].append(i)

    return kmeans, cluster_map

def add_distance(matrix, cluster_map):
    """
    Compute Euclidean distance for each element in a cluster to its mean.

    Parameters:
    - matrix (numpy.ndarray): Input matrix.
    - cluster_map (dict): Dictionary mapping clusters to indices.

    Returns:
    dict: Updated cluster map with distances.
    """
    for k, v in cluster_map.items():
        cluster_mean = matrix[v].mean(axis=0)
        result = [(i, sp_distance.euclidean(matrix[i], cluster_mean)) for i in v]
        cluster_map[k] = np.asarray(result)
    return cluster_map

def cluster_transmap(kmeans_cluster, dendrogram_cluster):
    """
    Map k-means clusters to dendrogram clusters based on minimum differences.

    Parameters:
    - kmeans_cluster (dict): Dictionary mapping k-means clusters to indices.
    - dendrogram_cluster (dict): Dictionary mapping dendrogram clusters to indices.

    Returns:
    dict: Remapped k-means clusters.
    """
    kmeans_remap = {}    
    # Iterate over k-means clusters
    for kmeans_key, kmeans_value in kmeans_cluster.items():
        # Find dendrogram cluster with minimum different elements
        dendrogram_key, dendrogram_value = min(
            dendrogram_cluster.items(),
            key=lambda x: len(set(kmeans_value).symmetric_difference(x[1]))
        )
        # Update mapping
        kmeans_remap.update({dendrogram_key: kmeans_value})
    return kmeans_remap