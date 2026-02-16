# AuGPatt-BiLSTM: Improving Extreme Streamflow Prediction

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and sample testing results for the manuscript:  
**"Improving Extreme Streamflow Prediction by Integrating CEEMDAN, Autoencoders, and Attention-Enhanced BiLSTM Optimized via GSA-PSO"**.

## 📖 Overview
Accurate prediction of extreme streamflow events (e.g., flood peaks) is crucial for water resource management and disaster mitigation. However, highly nonlinear and non-stationary daily streamflow series—especially in flashy micro-scale catchments—pose significant challenges for traditional data-driven models. 

To address this, we propose **AuGPatt-BiLSTM**, a novel hybrid deep learning framework integrating four advanced modules:
1. **CEEMDAN**: Decomposes the complex raw streamflow series into Intrinsic Mode Functions (IMFs).
2. **Autoencoder (Au)**: Efficiently extracts robust features and denoises high-frequency IMFs, preventing the loss of critical precursor information.
3. **GSA-PSO Algorithm**: A population-based heuristic algorithm (Gravitational Search Algorithm-based Particle Swarm Optimization) that automatically optimizes key hyperparameters without relying on subjective prior statistical assumptions.
4. **Attention Mechanism**: Dynamically assigns higher weights to critical historical time steps to mitigate "memory dilution" over long sequences, specifically enhancing the tracking of extreme events.

The framework has been rigorously validated across multiple spatial scales, from large mainstream catchments (Yellow River Basin, China) to medium and micro-scale catchments (US CAMELS dataset).

## 🗂️ Repository Structure

    AuGPatt-BiLSTM/
    │
    ├── Code/                    # Source code for the AuGPatt-BiLSTM framework and three ablation study variants
    │
    ├── Results/                 # Model simulation outputs vs. observed data (Testing phase)
    │   ├── Results of Animas River     # Medium-scale catchment (US CAMELS dataset)
    │   └── Results of Jacob Fork       # Micro-scale flashy catchment (US CAMELS dataset)
    │
    └── README.md                # Project documentation

## ⚙️ Requirements & Environment

The model was developed and tested using **Python 3.9**. The hardware utilized for training was an Intel(R) Core(TM) i7-9700 CPU (3.00 GHz) with 64 GB RAM and an NVIDIA GeForce RTX 3090 GPU.

**Core Dependencies:**
- `numpy`, `pandas`, `scikit-learn`
- `PyEMD` (for CEEMDAN implementation)
- `TensorFlow` / `PyTorch` *(Please choose based on your backend)*
- `matplotlib` (for hydrograph visualization)

**PIP Install Code:**

    pip install numpy pandas openpyxl torch scikit-learn statsmodels EMD-signal

## 🚀 Data Availability

The original daily streamflow data for the Yellow River Basin stations were obtained from the [National Earth System Science Data Center, China](https://www.geodata.cn). Due to data redistribution policies, the complete raw dataset is available upon reasonable request.

The US catchment data (Animas River and Jacob Fork) are publicly available from the open-source **CAMELS dataset**. Sample testing results to verify the model's outputs are provided in the `Results/` folder.

## 📝 Citation

If you find this code, data, or methodology useful in your research, please consider citing our paper once it is published.

## ✉️ Contact

For any questions, discussions, or data requests, please feel free to open an issue in this repository or contact the authors:
- **Xue Li**: xli@tjnu.edu.cn
- **Jian Sha**: shajian@tjnu.edu.cn
