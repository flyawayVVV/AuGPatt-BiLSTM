# AuGPatt-BiLSTM: Improving Extreme Streamflow Prediction

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and sample testing results for the manuscript:  
**"Improving Extreme Streamflow Prediction by Integrating CEEMDAN, Autoencoders, and Attention-Enhanced BiLSTM Optimized via GSA-PSO"**.

## 📖 Overview
Accurate prediction of extreme streamflow events (e.g., flood peaks and low flows) is crucial for water resource management and disaster mitigation. However, highly nonlinear and non-stationary daily streamflow series—especially in flashy micro-scale catchments—pose significant challenges for traditional data-driven models. 

To address this, we propose **AuGPatt-BiLSTM**, a novel hybrid deep learning framework integrating four advanced modules:
1. **CEEMDAN**: Decomposes the complex raw streamflow series into Intrinsic Mode Functions (IMFs).
2. **Autoencoder (Au)**: Efficiently extracts robust features and denoises high-frequency IMFs, preventing the loss of critical precursor information.
3. **GSA-PSO Algorithm**: A population-based heuristic algorithm (Gravitational Search Algorithm-based Particle Swarm Optimization) that automatically optimizes key hyperparameters without relying on subjective prior statistical assumptions.
4. **Attention Mechanism**: Dynamically assigns higher weights to critical historical time steps to mitigate "memory dilution" over long sequences, specifically enhancing the tracking of extreme events.

The framework has been rigorously validated across multiple spatial scales, from large mainstream catchments (Yellow River Basin, China) to medium and micro-scale catchments (US CAMELS dataset).

## 🗂️ Repository Structure
```text
AuGPatt-BiLSTM/
│
├── Code/                    # Source code for the AuGPatt-BiLSTM framework
│   ├── ...                  # (Please upload your Python scripts here)
│
├── Results/                 # Model simulation outputs vs. observed data (Testing phase)
│   ├── Tongguan.csv         # Mainstream catchment (Yellow River Basin)
│   ├── Huaxian.csv          # Tributary catchment (Yellow River Basin)
│   ├── Animas_River.csv     # Medium-scale catchment (US CAMELS dataset)
│   └── Jacob_Fork.csv       # Micro-scale flashy catchment (US CAMELS dataset)
│
└── README.md                # Project documentation
