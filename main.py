from pipeline import run_Duffing_problem_anomaly_mesure




if __name__ == "__main__":
    
    run_Duffing_problem_anomaly_mesure(
        beta_min=0.10,          # min value of beta
        beta_max=0.35,          # max value of beta
        num_beta=20,            # number of beta values to test between beta_min and beta_max
        nominal_beta=0.10,      # reference value of beta to consider as normal data
        A=22.0,                 # Driving amplitude of Duffing problem
        omega=5.0,              # Driving frequency of Duffing problem
        sampling_rate=100,      # sampling rate of the data
        total_time=40,          # Total duration of the time series in seconds
        models=['PCA', 'MLPNN', 'RBFNN', 'SFNN', 'WS'],  # List of models to run: ['PCA', 'MLPNN', 'RBFNN', 'SFNN', 'WS']
        alphabet_size=8,        # size of alphabet for symbolization
        D=1                     # order of the Markov Chain
    )
    