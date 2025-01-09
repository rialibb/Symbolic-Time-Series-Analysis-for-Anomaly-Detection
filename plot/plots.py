import matplotlib.pyplot as plt



def plot_models(anomaly_mesures):
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
    plt.grid()
    plt.legend()
    plt.savefig('plot/anomaly_comparison_plot.png', dpi=300, bbox_inches='tight')
    
    
    
    
    
    
def plot_symbolization_comparison_WS(ws_anomaly_measures_sax, ws_anomaly_measures_linear):
    """
    Plots the comparison between the linear and the SAX symbolization for WS method.
    Args:
        ws_anomaly_measures_sax(dict): Dictionary containing the anomaly measures for different beta values for SAX symbolization.
        ws_anomaly_measures_linear(dict): Dictionary containing the anomaly measures for different beta values for linear symbolization.
    """
    
    # Plot all anomaly measures
    plt.figure(figsize=(12, 8))
    
    plt.plot(list(ws_anomaly_measures_sax.keys()), list(ws_anomaly_measures_sax.values()), marker='o', label='SAX')
    plt.plot(list(ws_anomaly_measures_linear.keys()), list(ws_anomaly_measures_linear.values()), marker='x', label='Linear')
    
    plt.xlabel('Beta values (β)')
    plt.ylabel('Normalized Anomaly Measure')
    plt.title('Comparison of Anomaly Detection Methods for WS for different Symbolization techniques')
    plt.grid()
    plt.legend()
    plt.savefig('plot/Symbolization_comparison_plot.png', dpi=300, bbox_inches='tight')
    
    
    
    
    
    
    
    
def plot_Markov_order_comparison_for_WS(anomaly_mesures):
    """
    Plots the evolution of the anomaly measures over beta values for different Markov orders D for WS method on Duffing problem.
    Args:
        anomaly_mesures(dict): Dictionary containing the anomaly measures for each Markov order.
    """
    
    
    # Plot all anomaly measures
    plt.figure(figsize=(12, 8))
    
    for (order, mesure) in anomaly_mesures.items():
        plt.plot(list(mesure.keys()), list(mesure.values()), label=f'D={order}')
    
    plt.xlabel('Beta values (β)')
    plt.ylabel('Normalized Anomaly Measure')
    plt.title('Comparison of Anomaly Detection Methods for WS for different Markov orders D')
    plt.grid()
    plt.legend()
    plt.savefig('plot/Markov_orders_comparison_plot.png', dpi=300, bbox_inches='tight')








def plot_delay_influence(anomaly_mesures):
    """
    Plots the evolution of the anomaly measures over beta values for different models.
    Args:
        anomaly_mesures(dict): Dictionary containing the anomaly measures for each model.
    """
    
    # Plot all anomaly measures
    plt.figure(figsize=(12, 8))
    color = ['#ad0100', '#f60002', '#f74d4d', '#f79b9c', '#f6eaea']
    
    for i, (name, mesure) in enumerate(anomaly_mesures.items()):
        plt.plot(list(mesure.keys()), list(mesure.values()), marker='x', label=name, color=color[i])
    
    plt.xlabel('Delay values')
    plt.ylabel('Normalized Anomaly Measure')
    plt.title('Influence of delay in SFNN method')
    plt.grid()
    plt.legend()
    plt.savefig('plot/delay_influence.png', dpi=300, bbox_inches='tight')