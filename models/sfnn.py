import numpy as np
from models import d_markov_anomaly_measure




def symbolic_false_nearest_neighbors(data, alphabet_size):
    """Apply SFNN (Symbolic False Nearest Neighbours) for partitioning time series into symbolic sequences."""
    ######
    ###### TODO : implement this function
    ######
    bins = np.linspace(np.min(data), np.max(data)+1e-6, alphabet_size + 1)
    symbolic_sequence = np.digitize(data, bins, right=True) - 1
    return symbolic_sequence







def compute_sfnn_anomaly_scores(data_scaled, alphabet_size = 8, D=1, nominal_beta = 0.10):
    """
    Calculate the anomaly measure using the D-Markov Machine with Symbolic False Nearest Neighbours (SFNN).
    Args:
        data_scaled: Normalized data based on z_score method.
        alphabet_size: Size of the symbol alphabet.
        D: Order of the Markov machine.
        nominal_beta: The nominal value of beta to be considered as a reference for the Dugging problem.
    Returns:
        anomaly_measure: Anomaly measure based on stationary probability vector.
    """
    
    sfnn_anomaly_measures = {}

    for beta, data in data_scaled.items():
        
        # SFNN Symbolization
        symbolic_sfnn = symbolic_false_nearest_neighbors(data, alphabet_size=alphabet_size)
        nominal_sfnn = symbolic_false_nearest_neighbors(data_scaled[nominal_beta], alphabet_size=alphabet_size)

        # SFNN D-Markov Anomaly Measure
        sfnn_anomaly_measures[beta] = d_markov_anomaly_measure(symbolic_sfnn, nominal_sfnn, alphabet_size, D)

    # Normalize SFNN measures
    sfnn_anomaly_measures = {beta: value - sfnn_anomaly_measures[0.10] for beta, value in sfnn_anomaly_measures.items()}
    max_sfnn = max(sfnn_anomaly_measures.values())
    sfnn_anomaly_measures = {beta: value / max_sfnn for beta, value in sfnn_anomaly_measures.items()}
    
    return sfnn_anomaly_measures