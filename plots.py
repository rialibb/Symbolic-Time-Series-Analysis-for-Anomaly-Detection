import matplotlib.pyplot as plt



def plot_results(anomaly_mesures):
    """
    Plots the evolution of the anomaly measures over beta values for different models.
    Args:
        anomaly_mesures(dict): Dictionary containing the anomaly measures for each model.
    """
    
    markers = ['o', 'x', 's', 'd', '^']
    
    # Plot all anomaly measures
    plt.figure(figsize=(12, 8))
    
    for i, (name, mesure) in enumerate(anomaly_mesures.items()):
        plt.plot(list(mesure.keys()), list(mesure.values()), marker=markers[i], label=name)
    
    plt.xlabel('Beta values (β)')
    plt.ylabel('Normalized Anomaly Measure')
    plt.title('Comparison of Anomaly Detection Methods (Normalized)')
    plt.legend()
    plt.savefig('anomaly_comparison_plot.png', dpi=300, bbox_inches='tight')