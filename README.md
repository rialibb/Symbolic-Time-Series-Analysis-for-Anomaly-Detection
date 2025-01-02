# Symbolic Time Series Analysis for Anomaly Detection

This repository contains a Python implementation of the methods described in the paper **"Symbolic Time Series Analysis for Anomaly Detection: A Comparative Evaluation"** by [Shin C. Chin, Asok Ray, and Venkatesh Rajagopalan](https://www.me.psu.edu/ray/journalAsokRay/2005/159AnomalyPatternComparison.pdf). This implementation is an attempt to explore the power of symbolic time series analysis (STSA) in anomaly detection, specifically using the **D-Markov Machine**, a finite-state machine approach based on symbolic dynamics.

## Motivation

The paper introduces a novel anomaly detection technique based on symbolic time series analysis and compares it with other techniques such as PCA, MLPNNs, and RBFNNs. However, no Python implementation of this method was available at the time of this project. This repository provides a self-implemented version to evaluate the capabilities of STSA for anomaly detection, focusing on:

- Early detection of small anomalies.
- Robustness to measurement noise.
- Comparison with other pattern recognition techniques.

## Key Features

- **D-Markov Machine Implementation**: A symbolic dynamics-based anomaly detection method, incorporating:
  - Symbolic sequence generation using phase-space partitioning or wavelet-space partitioning.
  - Construction of a finite-state automaton for anomaly characterization.
- **Comparative Methods**: Includes PCA and neural network-based methods (MLPNN and RBFNN) for performance benchmarking.
- **Symbolization Techniques**:
  - Symbolic False Nearest Neighbors (SFNN) partitioning.
  - Wavelet Space (WS) partitioning.

## Contents

- **`models/d_markov_machine`**: Functions to construct the D_Markov States, generation of stationary probabilistic vector and calculation of the different anomaly mesures based on KL divergence.
- **`models/mlpnn`**: Compute the anomaly scores based on the MLPNN model.
- **`models/pca`**: Compute the anomaly scores based on the PCA model.
- **`models/rbfnn`**: Compute the anomaly scores based on the RBFNN model.
- **`models/sfnn`**: Compute the anomaly scores based on the D_Markov_Machine with SFNN model.
- **`models/ws`**: Compute the anomaly scores based on the D_Markov_Machine with WS model.
- **`duffing_data`**: Generate and preprocess time series based on Duffing Equation.
- **`main`**: Executable file to compute the anomaly detection models and compare the results.
- **`pipeline`**: Contains the pipeline of the probject.
- **`plots`**: Function to plot the anomay detection results.
- **`LICENSE`**: License agreement for the implementation.
- **`requirements.txt`**: List of required Python packages for the implementation.



## Prerequisites

- Python 3.8 or later
- Libraries:
  - `numpy`
  - `sklearn`
  - `matplotlib`
  - `pywt`
  - `saxpy`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rialibb/Symbolic-Time-Series-Analysis-for-Anomaly-Detection.git

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Results

* Comparative performance curves for anomaly measures, demonstrating the superior early detection capabilities of the D-Markov machine with SFNN and WS partitioning.
* Visualizations of symbolic sequences, finite-state automata, and anomaly measure trends.


## Contributions
Contributions, suggestions, and feedback are welcome! Feel free to open an issue or a pull request.

