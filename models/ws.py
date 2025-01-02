import numpy as np
import pywt
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import ts_to_string
from models import d_markov_anomaly_measure



# Wavelet Space Method with SAX symbolization
def wavelet_space_partitioning_sax(data, wavelet='db1', level=1, alphabet_size=8):
    """
    Partition the data in wavelet space using SAX symbolization.
    Args:
        data: The input time series data.
        wavelet: The type of wavelet to use for decomposition (default: 'db1').
        level: The level of wavelet decomposition to perform (default: 1).
        alphabet_size: The size of the SAX alphabet (default: 8).
    Returns:
        symbolic_sequence: A symbolic representation of the wavelet coefficients using SAX.
    """
    # Perform the wavelet decomposition at specified scale levels
    coeffs = pywt.wavedec(data, wavelet, level=level)  # Fixed level=1 as described in the paper
    
    # Flatten the wavelet coefficients
    flattened_coeffs = np.concatenate(coeffs)

    # Standardize the coefficients to have zero mean and unit variance
    standardized_coeffs = (flattened_coeffs - np.mean(flattened_coeffs)) / np.std(flattened_coeffs)

    # Generate SAX cuts for the specified alphabet size
    sax_cuts = cuts_for_asize(alphabet_size)

    # Apply SAX symbolization
    sax_symbols = ts_to_string(standardized_coeffs, sax_cuts)

    return np.array([ord(symbol) - ord('a') for symbol in sax_symbols])










def compute_ws_anomaly_scores(data_scaled, alphabet_size = 8, D=1, nominal_beta = 0.10):
    """
    Calculate the anomaly measure using the D-Markov Machine with Wavelet Space (WS) partitioning.
    Args:
        data_scaled: Normalized data based on z_score method.
        alphabet_size: Size of the symbol alphabet.
        D: Order of the Markov machine.
        nominal_beta: The nominal value of beta to be considered as a reference for the Dugging problem.
    Returns:
        anomaly_measure: Anomaly measure based on stationary probability vector.
    """
    
    ws_anomaly_measures = {}
    
    for beta, data in data_scaled.items():

        # WS Symbolization
        symbolic_ws = wavelet_space_partitioning_sax(data, wavelet='db1', alphabet_size=alphabet_size)
        nominal_ws = wavelet_space_partitioning_sax(data_scaled[nominal_beta], wavelet='db1', alphabet_size=alphabet_size)

        # WS D-Markov Anomaly Measure
        ws_anomaly_measures[beta] = d_markov_anomaly_measure(symbolic_ws, nominal_ws, alphabet_size, D)

    # Normalize WS measures
    ws_anomaly_measures = {beta: value - ws_anomaly_measures[nominal_beta] for beta, value in ws_anomaly_measures.items()}
    max_ws = max(ws_anomaly_measures.values())
    ws_anomaly_measures = {beta: value / max_ws for beta, value in ws_anomaly_measures.items()}
    
    return ws_anomaly_measures