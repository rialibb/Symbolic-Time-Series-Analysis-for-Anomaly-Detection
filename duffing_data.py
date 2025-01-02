import numpy as np
from sklearn.preprocessing import StandardScaler





# Simulate the nonlinear electronic system based on the Duffing equation
def generate_time_series(beta_values, A, omega, sampling_rate, total_time):
    """
    Generate synthetic time series data based on the Duffing oscillator.
    Args:
        beta_values: List of dissipation parameter values (beta).
        A: Amplitude of the driving force.
        omega: Frequency of the driving force.
        sampling_rate: Sampling rate for time series generation.
        total_time: Total duration of the time series.
    Returns:
        time_series_data: Dictionary mapping beta values to corresponding time series.
    """
    time = np.arange(0, total_time, 1 / sampling_rate)
    time_series_data = {}

    for beta in beta_values:
        y = np.zeros(len(time))
        dy_dt = np.zeros(len(time))
        
        for i in range(1, len(time)):
            d2y_dt2 = -beta * dy_dt[i - 1] - y[i - 1] - y[i - 1]**3 + A * np.cos(omega * time[i])
            dy_dt[i] = dy_dt[i - 1] + d2y_dt2 * (1 / sampling_rate)
            y[i] = y[i - 1] + dy_dt[i] * (1 / sampling_rate)

        time_series_data[beta] = y

    return time_series_data








def reshape_to_segments(data, segment_length=10):
    """
    Reshape data into a matrix with 270 segments of length 10.
    Args:
        data: Input data.
        segment_length: length of each segment.
    Returns:
        data: reshaped data as a matrix of length (num_samples, segment_length).
    """
    
    num_segments = len(data) // segment_length
    return data[:num_segments * segment_length].reshape((num_segments, segment_length))







def preprocess_time_series(beta_values, A, omega, sampling_rate, total_time):
    """
    preprocess the generated synthetic time series data based on the Duffing oscillator.
    Args:
        beta_values: List of dissipation parameter values (beta).
        A: Amplitude of the driving force.
        omega: Frequency of the driving force.
        sampling_rate: Sampling rate for time series generation.
        total_time: Total duration of the time series.
    Returns:
        data_scaled: Normalized data based on z_score method.
        segmented_data: Reshaped data into a matrix of segments.
    """
    
    # Generate time series data
    time_series_data = generate_time_series(beta_values, A, omega, sampling_rate, total_time)

    # Preprocess the data
    scaler = StandardScaler()
    data_scaled = {beta: scaler.fit_transform(time_series.reshape(-1, 1)).flatten() 
                for beta, time_series in time_series_data.items()}

    # Reshape data into a matrix with 270 segments of length 10
    segmented_data = {beta: reshape_to_segments(data, segment_length=10) 
                    for beta, data in data_scaled.items()}

    return data_scaled, segmented_data



