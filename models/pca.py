from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error




def compute_pca_anomaly_scores(segmented_data, nominal_beta=0.10):
    """
    Compute the anomaly scores based on the MLPNN model.
    Args:
        segmented_data: The input data segmented into a matrix of (num_samples, sample_length).
        nominal_beta: The nominal value of beta to be considered as a reference for the Dugging problem.
    Returns:
        pca_anomaly_measures: The anomaly scores based on different values of beta
    """
    
    # Apply PCA to segmented data
    pca_results = {}
    for beta, segments in segmented_data.items():
        pca = PCA(n_components=2)  # Assuming at least 2 features in reshaped data
        pca_results[beta] = pca.fit_transform(segments)

    # Use PCA results for anomaly detection
    pca_anomaly_measures = {}
    for beta, transformed_segments in pca_results.items():
        nominal_transformed = pca_results[nominal_beta]  # Nominal beta transformed segments
        mse = mean_squared_error(nominal_transformed.flatten(), transformed_segments.flatten())
        pca_anomaly_measures[beta] = mse

    # Normalize PCA anomaly measures
    pca_anomaly_measures = {beta: value - pca_anomaly_measures[nominal_beta] for beta, value in pca_anomaly_measures.items()}
    max_pca = max(pca_anomaly_measures.values())
    pca_anomaly_measures = {beta: value / max_pca for beta, value in pca_anomaly_measures.items()}
    
    # return the anomaly scores
    return pca_anomaly_measures