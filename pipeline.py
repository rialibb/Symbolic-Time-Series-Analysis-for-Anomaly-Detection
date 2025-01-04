import numpy as np
from duffing_data import preprocess_time_series
from models import (compute_mlpnn_anomaly_scores, 
                    compute_pca_anomaly_scores, 
                    compute_rbfnn_anomaly_scores, 
                    compute_sfnn_anomaly_scores, 
                    compute_ws_anomaly_scores)
from plot.plots import (plot_models, 
                        plot_symbolization_comparison_WS, 
                        plot_Markov_order_comparison_for_WS,
                        plot_delay_influence)





def run_Duffing_problem_anomaly_mesure(beta_min=0.10, 
                                       beta_max=0.35,
                                       num_beta=20,
                                       nominal_beta=0.10,
                                       A=22.0, 
                                       omega=5.0, 
                                       sampling_rate=100,
                                       total_time=40,
                                       models=['PCA', 'MLPNN', 'RBFNN', 'SFNN', 'WS'],
                                       alphabet_size=8, 
                                       D=1,
                                       threshold=0.95,
                                       symbolization='SAX'):
    """
    Run the different anomaly detection models for the Duffing problem
    Args:
        beta_min (float): The minimum value of beta.
        beta_max (float): The maximum value of beta.
        num_beta (float): The number of beta values to test between beta_min and beta_max.
        nominal_beta (float): The reference value of beta to consider as normal data.
        A (float): The Driving amplitude of the stimulus.
        omega (float): The driving frequency of the stimulus.
        sampling_rate (float): The sampling rate of the data.
        total_time (float): Total duration of the time series in seconds.
        models (list): List of models to run.
        alphabet_size (int): The size of the alphabet used to encode the time series into different symbols.
        D (int): The order of the Markov Chain.
        threshold (float): The cumulative variance threshold for selecting principal components.
        symbolization (str): type of symbolization (linear or SAX)
    """
    
    # compute beta values for the experiments
    beta_values = np.linspace(beta_min, beta_max, num_beta)
    
    # generate and preprocess time series based on Duffing problem
    data_scaled, segmented_data = preprocess_time_series(beta_values, A, omega, sampling_rate, total_time)
    
    # calculate the anomaly mesures for different models:
    anomaly_mesures={}
    
    for model in models:
        
        if model == 'PCA':
            pca_anomaly_measures = compute_pca_anomaly_scores(segmented_data, nominal_beta=nominal_beta, threshold=threshold)
            anomaly_mesures[model] = pca_anomaly_measures
        
        elif model == 'MLPNN':
            mlp_anomaly_measures = compute_mlpnn_anomaly_scores(segmented_data, nominal_beta=nominal_beta)
            anomaly_mesures[model] = mlp_anomaly_measures
            
        elif model == 'RBFNN':
            rbf_anomaly_measures = compute_rbfnn_anomaly_scores(segmented_data, nominal_beta=nominal_beta)
            anomaly_mesures[model] = rbf_anomaly_measures
            
        elif model == 'SFNN':
            sfnn_anomaly_measures = compute_sfnn_anomaly_scores(data_scaled, alphabet_size=alphabet_size, D=D, nominal_beta=nominal_beta)
            anomaly_mesures[model] = sfnn_anomaly_measures
            
        elif model == 'WS':
            ws_anomaly_measures = compute_ws_anomaly_scores(data_scaled, alphabet_size=alphabet_size, D=D, nominal_beta=nominal_beta, symbolization=symbolization)
            anomaly_mesures[model] = ws_anomaly_measures   
            
    # generate the plot of the anomaly measures for the different models
    plot_models(anomaly_mesures)
            
    
    
    
    
    
    
    
    
    
    
def run_Symbolization_comparison_for_WS(beta_min=0.10, 
                                       beta_max=0.35,
                                       num_beta=20,
                                       nominal_beta=0.10,
                                       A=22.0, 
                                       omega=5.0, 
                                       sampling_rate=100,
                                       total_time=40,
                                       alphabet_size=8, 
                                       D=1):
    """
    Run a comparison between Linear and SAX symbolization methods for the WS model for the Duffing problem.
    Args:
        beta_min (float): The minimum value of beta.
        beta_max (float): The maximum value of beta.
        num_beta (float): The number of beta values to test between beta_min and beta_max.
        nominal_beta (float): The reference value of beta to consider as normal data.
        A (float): The Driving amplitude of the stimulus.
        omega (float): The driving frequency of the stimulus.
        sampling_rate (float): The sampling rate of the data.
        total_time (float): Total duration of the time series in seconds.
        alphabet_size (int): The size of the alphabet used to encode the time series into different symbols.
        D (int): The order of the Markov Chain.
    """
    
    # compute beta values for the experiments
    beta_values = np.linspace(beta_min, beta_max, num_beta)
    
    # generate and preprocess time series based on Duffing problem
    data_scaled, _ = preprocess_time_series(beta_values, A, omega, sampling_rate, total_time)
    
    # compute anomaly scores for WS based on different symbolization techniques
    ws_anomaly_measures_sax = compute_ws_anomaly_scores(data_scaled, alphabet_size=alphabet_size, D=D, nominal_beta=nominal_beta, symbolization='SAX')
    ws_anomaly_measures_linear = compute_ws_anomaly_scores(data_scaled, alphabet_size=alphabet_size, D=D, nominal_beta=nominal_beta, symbolization='Linear')
    
    # generate the plot of the anomaly measures for the different symbolization techniques
    plot_symbolization_comparison_WS(ws_anomaly_measures_sax, ws_anomaly_measures_linear)

    
    
    
    
    
    
    
    
def run_Markov_order_comparison_for_WS(beta_min=0.10, 
                                       beta_max=0.35,
                                       num_beta=20,
                                       nominal_beta=0.10,
                                       A=22.0, 
                                       omega=5.0, 
                                       sampling_rate=100,
                                       total_time=40,
                                       alphabet_size=8, 
                                       D_values=[1,3,5],
                                       symbolization='SAX'):
    """
    Run a comparison between different Markov Chain orders for the WS model for the Duffing problem.
    Args:
        beta_min (float): The minimum value of beta.
        beta_max (float): The maximum value of beta.
        num_beta (float): The number of beta values to test between beta_min and beta_max.
        nominal_beta (float): The reference value of beta to consider as normal data.
        A (float): The Driving amplitude of the stimulus.
        omega (float): The driving frequency of the stimulus.
        sampling_rate (float): The sampling rate of the data.
        total_time (float): Total duration of the time series in seconds.
        alphabet_size (int): The size of the alphabet used to encode the time series into different symbols.
        D_values (list of int): The list of orders of the Markov Chain.
        symbolization (str): type of symbolization (linear or SAX)
    """
    
    # compute beta values for the experiments
    beta_values = np.linspace(beta_min, beta_max, num_beta)
    
    # generate and preprocess time series based on Duffing problem
    data_scaled, _ = preprocess_time_series(beta_values, A, omega, sampling_rate, total_time)
    
    # compute anomaly scores for WS based on different Markov orders
    anomaly_mesures={}
    
    for D in D_values:
        anomaly_mesures[D] = compute_ws_anomaly_scores(data_scaled, alphabet_size=alphabet_size, D=D, nominal_beta=nominal_beta, symbolization=symbolization)
    
    # generate the plot of the anomaly measures for the different symbolization techniques
    plot_Markov_order_comparison_for_WS(anomaly_mesures)



def delay_influence(beta_min=0.10, 
                    beta_max=0.35,
                    num_beta=20,
                    nominal_beta=0.10,
                    A=22.0, 
                    omega=5.0, 
                    sampling_rate=100,
                    total_time=40,
                    alphabet_size=8):


    # compute beta values for the experiments
    beta_values = np.linspace(beta_min, beta_max, num_beta)
    
    # generate and preprocess time series based on Duffing problem
    data_scaled, _ = preprocess_time_series(beta_values, A, omega, sampling_rate, total_time)

    anomaly_mesures={}

    delays = [8, 12, 18, 21, 30]
    for delay in delays:
        sfnn_anomaly_measures = compute_sfnn_anomaly_scores(data_scaled, alphabet_size=alphabet_size, D=1, nominal_beta=nominal_beta, delay=delay)
        anomaly_mesures[delay] = sfnn_anomaly_measures
    
    plot_delay_influence(anomaly_mesures)
