import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF




def compute_rbfnn_anomaly_scores(segmented_data, nominal_beta=0.10):
    """
    Compute the anomaly scores based on the MLPNN model.
    Args:
        segmented_data: The input data segmented into a matrix of (num_samples, sample_length).
        nominal_beta: The nominal value of beta to be considered as a reference for the Dugging problem.
    Returns:
        rbf_anomaly_measures: The anomaly scores based on different values of beta
    """
    
    # Implement RBFNN for anomaly detection
    nominal_segments = segmented_data[nominal_beta]
    X_nom = nominal_segments
    y_nom = np.mean(nominal_segments, axis=1)  

    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gpr.fit(X_nom, y_nom)

    rbf_anomaly_measures = {}
    for beta, segments in segmented_data.items():
        X = segments
        y = np.mean(segments, axis=1)
        predictions, _ = gpr.predict(X, return_std=True)
        mse = mean_squared_error(y, predictions)
        rbf_anomaly_measures[beta] = mse

    # Normalize RBF anomaly measures
    rbf_anomaly_measures = {beta: value - rbf_anomaly_measures[nominal_beta] for beta, value in rbf_anomaly_measures.items()}
    max_rbf = max(rbf_anomaly_measures.values())
    rbf_anomaly_measures = {beta: value / max_rbf for beta, value in rbf_anomaly_measures.items()}
    
    # return the anomaly scores
    return rbf_anomaly_measures