import numpy as np




# Helper function for D-Markov Machine state construction
def construct_d_markov_states(data, alphabet_size, D):
    """
    Constructs states for the D-Markov Machine and returns the transition matrix and state index mapping.
    Args:
        data: The input symbolic time series data.
        alphabet_size: Size of the symbol alphabet.
        D: The order of the Markov machine.
    Returns:
        transition_matrix: Transition matrix of size (n_states, n_states).
        state_index: Dictionary mapping states to their indices.
    """
    states = {}

    # Generate states for the D-Markov Machine
    for i in range(len(data) - D):
        state = tuple(data[i:i + D])
        next_symbol = data[i + D]
        if state not in states:
            states[state] = np.zeros(alphabet_size)
        states[state][next_symbol] += 1

    # Create a mapping of states to indices
    all_states = list(states.keys())
    state_index = {state: idx for idx, state in enumerate(all_states)}
    n_states = len(all_states)

    # Construct the transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    for state, transitions in states.items():
        row_idx = state_index[state]
        for next_symbol, prob in enumerate(transitions):
            next_state = tuple(list(state[1:]) + [next_symbol])
            if next_state in state_index:
                col_idx = state_index[next_state]
                transition_matrix[row_idx, col_idx] = prob / np.sum(transitions)

    return transition_matrix, state_index








# Helper function to compute the stationary probability vector
def compute_stationary_vector(transition_matrix):
    """
    Compute the stationary probability vector for a D-Markov Machine.
    Args:
        transition_matrix: Transition matrix of the D-Markov Machine.
    Returns:
        stationary_vector: The stationary probability vector.
    """
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    stationary_vector = eigvecs[:, np.isclose(eigvals, 1, atol=1e-6)].flatten().real
    stationary_vector /= np.sum(stationary_vector)  # Normalize

    return stationary_vector









# Implement the D-Markov Machine for SFNN and WS
def d_markov_anomaly_measure(symbolic_data, reference_data, alphabet_size, D):
    """
    Calculate the anomaly measure using the D-Markov Machine.
    Args:
        symbolic_data: Symbolic sequence for the test condition.
        reference_data: Symbolic sequence for the nominal condition.
        alphabet_size: Size of the symbol alphabet.
        D: Order of the Markov machine.
    Returns:
        anomaly_measure: Anomaly measure based on stationary probability vector.
    """
    reference_matrix, reference_index = construct_d_markov_states(reference_data, alphabet_size, D)
    test_matrix, test_index = construct_d_markov_states(symbolic_data, alphabet_size, D)

    # Ensure state indices align
    all_states = sorted(set(reference_index.keys()).union(test_index.keys()))
    ref_aligned = np.zeros((len(all_states), len(all_states)))
    test_aligned = np.zeros((len(all_states), len(all_states)))

    for state in all_states:
        if state in reference_index:
            ref_row = reference_matrix[reference_index[state], :]
            for next_state in all_states:
                if next_state in reference_index:
                    ref_aligned[all_states.index(state), all_states.index(next_state)] = ref_row[reference_index[next_state]]

        if state in test_index:
            test_row = test_matrix[test_index[state], :]
            for next_state in all_states:
                if next_state in test_index:
                    test_aligned[all_states.index(state), all_states.index(next_state)] = test_row[test_index[next_state]]

    # Normalize rows to ensure they sum to 1
    ref_aligned = np.nan_to_num(ref_aligned)
    row_sums = ref_aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for empty rows
    ref_aligned /= row_sums

    test_aligned = np.nan_to_num(test_aligned)
    row_sums = test_aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for empty rows
    test_aligned /= row_sums

    # Compute stationary probability vectors
    p_nom = compute_stationary_vector(ref_aligned)
    p_test = compute_stationary_vector(test_aligned)

    # Calculate anomaly measure based on the KL divergence
    kl_divergence = np.sum(p_nom * np.log((p_nom + 1e-8) / (p_test + 1e-8)))

    return kl_divergence