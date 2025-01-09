from pipeline import (run_Duffing_problem_anomaly_mesure,
                      run_Symbolization_comparison_for_WS,
                      run_Markov_order_comparison_for_WS,
                      run_delay_influence)




if __name__ == "__main__":
    
    #run_Duffing_problem_anomaly_mesure(
    #    beta_min=0.10,          # min value of beta
    #    beta_max=0.35,          # max value of beta
    #    num_beta=20,            # number of beta values to test between beta_min and beta_max
    #    nominal_beta=0.10,      # reference value of beta to consider as normal data
    #    A=22.0,                 # Driving amplitude of Duffing problem
    #    omega=5.0,              # Driving frequency of Duffing problem
    #    sampling_rate=100,      # sampling rate of the data
    #    total_time=40,          # Total duration of the time series in seconds
    #    models=['PCA', 'MLPNN', 'RBFNN', 'SFNN', 'WS'],  # List of models to run: ['PCA', 'MLPNN', 'RBFNN', 'SFNN', 'WS']
    #    alphabet_size=8,        # size of alphabet for symbolization
    #    D=1,                    # order of the Markov Chain
    #    threshold=0.95,         # cumulative variance threshold for selecting principal components
    #    symbolization='SAX'     # type of symbolization (linear or SAX)
    #)
    
    
    #run_delay_influence(
    #     beta_min=0.10,        # min value of beta
    #     beta_max=0.35,        # max value of beta
    #     num_beta=20,          # number of beta values to test between beta_min and beta_max
    #     nominal_beta=0.10,    # reference value of beta to consider as normal data
    #     A=22.0,               # Driving amplitude of Duffing problem
    #     omega=5.0,            # Driving frequency of Duffing problem
    #     sampling_rate=100,    # sampling rate of the data
    #     total_time=40,        # Total duration of the time series in seconds
    #     alphabet_size=8       # size of alphabet for symbolization
    #)
    
    
    #run_Symbolization_comparison_for_WS(
    #    beta_min=0.10,          # min value of beta
    #    beta_max=0.35,          # max value of beta
    #    num_beta=20,            # number of beta values to test between beta_min and beta_max
    #    nominal_beta=0.10,      # reference value of beta to consider as normal data
    #    A=22.0,                 # Driving amplitude of Duffing problem
    #    omega=5.0,              # Driving frequency of Duffing problem
    #    sampling_rate=100,      # sampling rate of the data
    #    total_time=40,          # Total duration of the time series in seconds
    #   alphabet_size=8,        # size of alphabet for symbolization
    #    D=1,                    # order of the Markov Chain
    #)
    
    
    #run_Markov_order_comparison_for_WS(
    #    beta_min=0.10,          # min value of beta
    #    beta_max=0.35,          # max value of beta
    #    num_beta=20,            # number of beta values to test between beta_min and beta_max
    #    nominal_beta=0.10,      # reference value of beta to consider as normal data
    #    A=22.0,                 # Driving amplitude of Duffing problem
    #    omega=5.0,              # Driving frequency of Duffing problem
    #    sampling_rate=100,      # sampling rate of the data
    #    total_time=40,          # Total duration of the time series in seconds
    #    alphabet_size=8,        # size of alphabet for symbolization
    #    D_values=[1,2,3,4,5],   # The list of orders of the Markov Chain
    #    symbolization='SAX'     # type of symbolization (linear or SAX)
    #)