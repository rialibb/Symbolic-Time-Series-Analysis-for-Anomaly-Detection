from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np








def compute_pca_anomaly_scores(segmented_data, nominal_beta=0.10, threshold=0.95):
    """
    Compute the anomaly scores based on the PCA method.
    Args:
        segmented_data: The input data segmented into a matrix of (num_samples, sample_length).
        nominal_beta: The nominal value of beta to be considered as a reference for the Duffing problem.
        threshold: The cumulative variance threshold for selecting principal components.
    Returns:
        pca_anomaly_measures: The anomaly scores based on different values of beta.
    """

    # Apply PCA to segmented data
    pca_results = {}
    for beta, segments in segmented_data.items():
        # Compute the covariance matrix
        covariance_matrix = np.cov(segments, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select q eigenvectors based on the cumulative variance threshold
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        q = np.searchsorted(cumulative_variance, threshold) + 1
        selected_eigenvectors = eigenvectors[:, :q]

        # Compute the normalized feature matrix
        normalization_factors = np.sqrt(eigenvalues[:q] / np.sum(eigenvalues))
        normalized_matrix = selected_eigenvectors * normalization_factors

        pca_results[beta] = normalized_matrix

    # Use PCA results for anomaly detection
    pca_anomaly_measures = {}
    for beta, normalized_matrix in pca_results.items():
        nominal_matrix = pca_results[nominal_beta]  # Nominal beta normalized matrix
        mse = mean_squared_error(nominal_matrix.flatten(), normalized_matrix.flatten())
        pca_anomaly_measures[beta] = mse

    # Normalize PCA anomaly measures
    pca_anomaly_measures = {beta: value - pca_anomaly_measures[nominal_beta] for beta, value in pca_anomaly_measures.items()}
    max_pca = max(pca_anomaly_measures.values())
    pca_anomaly_measures = {beta: value / max_pca for beta, value in pca_anomaly_measures.items()}

    # Return the anomaly scores
    return pca_anomaly_measures
