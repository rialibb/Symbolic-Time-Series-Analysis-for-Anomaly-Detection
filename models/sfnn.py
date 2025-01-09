import numpy as np
from models import d_markov_anomaly_measure
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def symbolic_false_nearest_neighbors(data, alphabet_size, centers=None, delay=18):
    # Compute the embedded with dimension d
    if centers is None: # Compute the centroid
        for dim in range(2,8):
            # Compute the embedded with dimension d
            n = len(data)
            
            embedded = np.zeros((n - (dim - 1) * delay, dim))
            for i in range(dim):
                embedded[:, i] = data[i * delay:n - (dim - 1 - i) * delay]

            # Find clusters and gives labels
            kmeans = KMeans(n_clusters=alphabet_size, random_state=42)
            labels = kmeans.fit_predict(embedded)
            centers = kmeans.cluster_centers_

            # Find the False Nearest Neighbor
            nbrs = NearestNeighbors(n_neighbors=2).fit(embedded)
            distances, indices = nbrs.kneighbors(embedded)
            false_neighbor = np.count_nonzero(labels != labels[indices[:,1]])
            false_neighbor /= len(labels)
            if false_neighbor < 0.01:
                break
    else:
        n = len(data)
        _, dim = centers.shape
        embedded = np.zeros((n - (dim - 1) * delay, dim))
        for i in range(dim):
            embedded[:, i] = data[i * delay:n - (dim - 1 - i) * delay]
        nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
        distances, labels = nbrs.kneighbors(embedded)
    return labels.flatten(), centers



def symbolic_false_nearest_neighbors_2(data, alphabet_size, d=3, centers=None, delay=18):
    """Apply SFNN (Symbolic False Nearest Neighbours) for partitioning time series into symbolic sequences.
    parameter to optimize: [alphabet_size*()]"""

    if centers is None:
        num_centers = alphabet_size

        # Setting bounds
        min_value, max_value = np.min(data), np.max(data)
        bound = []
        for i in range(num_centers):
            for _ in range(d):
                bound.append((min_value, max_value)) # Centers
            bound.append((0.1,1)) # Sigma

        # Optimization
        embedded = time_delay_embedding(data, delay, d)
        result = differential_evolution(objective_function, bound, args=(embedded, d, delay),
                                        disp=True,
                                        maxiter=50,

                                        workers=-1,
                                        updating='deferred',
                                        polish=False)


        # Fetch Result
        centers = np.zeros((num_centers, d))
        optimize_sigma = np.zeros(num_centers)
        for i in range(num_centers):
            for j in range(d):
                centers[i,j] = result.x[i*(d+1)+j]
            optimize_sigma[i] = result.x[i*(d+1)+d]

        # Compute label
        rbf_values = rbf_function(embedded, centers, optimize_sigma)
        labels = np.argmax(rbf_values, axis=1)
    else:
        embedded = time_delay_embedding(data, delay, d)

        # Fetch Result
        result = centers
        num_centers = alphabet_size
        optimize_centers = np.zeros((num_centers, d))
        optimize_sigma = np.zeros(num_centers)
        for i in range(num_centers):
            for j in range(d):
                optimize_centers[i,j] = result.x[i*(d+1)+j]
            optimize_sigma[i] = result.x[i*(d+1)+d]

        # Compute label
        rbf_values = rbf_function(embedded, optimize_centers, optimize_sigma)
        labels = np.argmax(rbf_values, axis=1)
    return labels.flatten(), result


def time_delay_embedding(series, delay, dim):
    """Reconstruct the phase space using time delay embedding.
    Args:
        series (np.ndarray): Time series data.
        delay (int): Time delay for embedding.
        dim (int): Embedding dimension.
    Returns:
        np.ndarray: Reconstructed phase space with the given embedding.
    """
    n = len(series)
    delay = int(delay)
    embedded = np.zeros((n - (dim - 1) * delay, dim))
    for i in range(dim):
        embedded[:, i] = series[i * delay:n - (dim - 1 - i) * delay]
    return embedded


def rbf_function(x, centers, sigma):
    """Return the RBF distance of x from centers"""
    distances = cdist(x, centers, 'euclidean')
    return np.exp(-(distances / sigma) ** 2)

def compute_false_nearest_neighbors(embedded, centers, sigma):
    """Compute the proportion of false nearest neighbors according to a RBF distance
    Args:
        embedded: state space series
        labels: symbol affected by the differential evolution algorithm
        centers:
        """
    # Partition according RBF
    rbf_values = rbf_function(embedded, centers, sigma)
    label = np.argmax(rbf_values, axis=1)

    # Find the Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embedded)
    _, indices = nbrs.kneighbors(embedded) #

    # Count how many closest Neighbors haven't the same label
    false_neigh_count = np.sum(label != label[indices[:,1]])
    return false_neigh_count / len(label)


def objective_function(params, embedded, d, delay):
    """Objective function for the differential evolution algorithm
    Args:
        params: centers, sigmas concatenated for the algorithm
        data: series
    Returns:
        """
    num_centers = (len(params) - 1) // (d + 1)
    centers = np.zeros((num_centers, d))
    sigma = np.zeros(num_centers)
    for i in range(num_centers):
        for j in range(d):
            centers[i,j] = params[i*(d+1)+j]
        sigma[i] = params[i*(d+1)+d]
    false_neigh_count = compute_false_nearest_neighbors(embedded, centers, sigma)

    # Regularisation
    penalty = 0
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            dist = np.linalg.norm(centers[i] - centers[j])
            penalty += 1e-3 / (dist + 1e-6)

    return false_neigh_count + penalty









def compute_sfnn_anomaly_scores(data_scaled, alphabet_size = 8, D=1, nominal_beta = 0.10, delay=18):
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
    nominal_sfnn, centers = symbolic_false_nearest_neighbors(data_scaled[nominal_beta], alphabet_size=alphabet_size, delay=delay)

    for beta, data in data_scaled.items():

        # SFNN Symbolization
        symbolic_sfnn, _ = symbolic_false_nearest_neighbors(data, alphabet_size=alphabet_size, centers = centers, delay=delay)
        if beta==0.1:
            symbolic_sfnn = nominal_sfnn

        # SFNN D-Markov Anomaly Measure
        sfnn_anomaly_measures[beta] = d_markov_anomaly_measure(symbolic_sfnn, nominal_sfnn, alphabet_size, D)

    # Normalize SFNN measures
    sfnn_anomaly_measures = {beta: value - sfnn_anomaly_measures[0.10] for beta, value in sfnn_anomaly_measures.items()}
    max_sfnn = max(sfnn_anomaly_measures.values())
    sfnn_anomaly_measures = {beta: value / max_sfnn for beta, value in sfnn_anomaly_measures.items()}

    return sfnn_anomaly_measures
