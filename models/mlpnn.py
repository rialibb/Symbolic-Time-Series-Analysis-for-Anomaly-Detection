import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error




def compute_mlpnn_anomaly_scores(segmented_data, nominal_beta=0.10):
    """
    Compute the anomaly scores based on the MLPNN model.
    Args:
        segmented_data: The input data segmented into a matrix of (num_samples, sample_length).
        nominal_beta: The nominal value of beta to be considered as a reference for the Dugging problem.
    Returns:
        mlp_anomaly_measures: The anomaly scores based on different values of beta
    """
    
    # Train MLP on nominal condition (beta = 0.10)
    nominal_segments = segmented_data[nominal_beta]
    X_nom = nominal_segments
    y_nom = np.mean(nominal_segments, axis=1)  

    mlp = MLPRegressor(hidden_layer_sizes=(50, 40, 30, 40), max_iter=1000, tol=1e-5, 
                        activation='tanh', random_state=42)
    mlp.fit(X_nom, y_nom)

    # Evaluate anomaly measures for MLP
    mlp_anomaly_measures = {}
    for beta, segments in segmented_data.items():
        X = segments
        y = np.mean(segments, axis=1)
        predictions = mlp.predict(X)
        mse = mean_squared_error(y, predictions)
        mlp_anomaly_measures[beta] = mse

    # Normalize MLP anomaly measures
    mlp_anomaly_measures = {beta: value - mlp_anomaly_measures[nominal_beta] for beta, value in mlp_anomaly_measures.items()}
    max_mlp = max(mlp_anomaly_measures.values())
    mlp_anomaly_measures = {beta: value / max_mlp for beta, value in mlp_anomaly_measures.items()}
    
    # return the anomaly scores
    return mlp_anomaly_measures