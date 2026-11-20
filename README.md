# Deep Learning (CSA4702) - Laboratory Experiments

A comprehensive collection of **30 Deep Learning Laboratory Experiments** organized by Course Outcomes (CO1 to CO5), featuring complete Python code, analytical output tables, and publication-quality visualization plots.

Repository URL: [https://github.com/Tamilarasan13092005/Deep-learning](https://github.com/Tamilarasan13092005/Deep-learning)

---

## 📁 Repository Structure

```
Deep-learning/
├── Assigments/                                    # Course Assignment Documents (CO1 - CO5)
│   ├── CSA47_A1_CO1_Tamilarasan_M.docx
│   ├── CSA47_A2_CO2_Tamilarasan_M.docx
│   ├── CSA47_A3_CO3_Tamilarasan_M.docx
│   ├── CSA47_A4_CO4_Tamilarasan_M.docx
│   └── CSA47_A5_CO5_Tamilarasan_M.docx
├── Experiments/                                   # All 30 Lab Experiments with Plots & Tables
│   ├── Exp. 1/                                    # Vector & Matrix Operations (p1.py, plots, tables)
│   ├── Exp. 2/                                    # Covariance, Eigenvalues & Eigenvectors
│   ├── Exp. 3/                                    # Probability Distributions & Bayes' Theorem
│   ├── Exp. 4/                                    # Entropy, Cross-Entropy & KL Divergence
│   ├── Exp. 5/                                    # Bias–Variance, Overfitting & Underfitting
│   ├── Exp. 6/                                    # Gradient Descent, SGD, Momentum & Adam
│   ├── Exp. 7/                                    # Perceptron & Forward Propagation
│   ├── Exp. 8/                                    # Two-Layer Neural Network with Backpropagation
│   ├── Exp. 9/                                    # Multilayer Perceptron (MLP)
│   ├── Exp. 10/                                   # L2, Dropout & Early Stopping Comparison
│   ├── Exp. 11/                                   # CNN for Image Classification
│   ├── Exp. 12/                                   # CNN Architecture & Hyperparameter Tuning
│   ├── Exp. 13/                                   # Sequential Dataset Preparation & RNN
│   ├── Exp. 14/                                   # Unfolded RNN Architecture Visualization
│   ├── Exp. 15/                                   # BPTT Flow & Gradient Analysis
│   ├── Exp. 16/                                   # LSTM vs RNN Performance Comparison
│   ├── Exp. 17/                                   # RNN Gradient Analysis & Gradient Clipping
│   ├── Exp. 18/                                   # RNN, LSTM & GRU Hyperparameter Tuning
│   ├── Exp. 19/                                   # PCA & Explained Variance Analysis
│   ├── Exp. 20/                                   # Probabilistic PCA & Reconstruction Analysis
│   ├── Exp. 21/                                   # Factor Analysis & Latent Factor Interpretation
│   ├── Exp. 22/                                   # Independent Component Analysis (ICA)
│   ├── Exp. 23/                                   # Bayesian Network Construction & Visualization
│   ├── Exp. 24/                                   # Graphical Model Sampling & Probabilistic Inference
│   ├── Exp. 25/                                   # RBM Energy & Reconstruction
│   ├── Exp. 26/                                   # Gibbs Sampling & Contrastive Divergence (CD-k)
│   ├── Exp. 27/                                   # Deep Belief Network (DBN) Representation Learning
│   ├── Exp. 28/                                   # Deep Boltzmann Machine (DBM) Simulation
│   ├── Exp. 29/                                   # VAE / GAN Sample Generation
│   └── Exp. 30/                                   # Generative Model Output Comparison & Evaluation
└── README.md
```

---

## 📋 List of Experiments & Artifacts

| S.No | CO | Experiment Title | Script | Plots & Visualizations | Output Tables |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **1** | CO1 | Vector and Matrix Operations for Deep Learning | [`p1.py`](Experiments/Exp.%201/p1.py) | `plot1_operations.png`, `plot2_histogram.png` | Tables 1–7 (Matrices, Stats & Verifications) |
| **2** | CO1 | Covariance, Eigenvalues and Eigenvectors Analysis | [`p2.py`](Experiments/Exp.%202/p2.py) | `plot1_scatter.png`, `plot2_covariance_map.png` | Tables 1–6 (Covariance, Eigendecomposition) |
| **3** | CO1 | Probability Distributions and Bayes’ Theorem | [`p3.py`](Experiments/Exp.%203/p3.py) | `plot1_distributions.png`, `plot2_bayesian.png` | Tables 1–6 (Distributions & Bayes Analysis) |
| **4** | CO1 | Entropy, Cross-Entropy and KL Divergence | [`p4.py`](Experiments/Exp.%204/p4.py) | `plot1_entropy_curves.png`, `plot2_kl_map.png` | Tables 1–5 (Entropy & Divergence Measures) |
| **5** | CO1 | Bias–Variance, Overfitting and Underfitting Analysis | [`p5.py`](Experiments/Exp.%205/p5.py) | `plot1_polynomial_fits.png`, `plot2_curves.png` | Tables 1–5 (Model Complexity & Errors) |
| **6** | CO1 | Gradient Descent, SGD, Momentum and Adam Comparison | [`p6.py`](Experiments/Exp.%206/p6.py) | `plot1_loss_surfaces.png`, `plot2_convergence.png`| Tables 1–5 (Optimizer Trajectories & Steps) |
| **7** | CO2 | Perceptron and Forward Propagation | [`p7.py`](Experiments/Exp.%207/p7.py) | `plot1_decision_boundary.png`, `plot2_loss.png` | Tables 1–4 (Forward Prop & Epoch History) |
| **8** | CO2 | Two-Layer Neural Network with Backpropagation | [`p8.py`](Experiments/Exp.%208/p8.py) | `plot1_xor_boundary.png`, `plot2_training.png` | Tables 1–5 (Weight Updates & Losses) |
| **9** | CO2 | Multilayer Perceptron Implementation | [`p9.py`](Experiments/Exp.%209/p9.py) | `plot1_accuracy_loss.png`, `plot2_confusion.png`| Tables 1–5 (Architecture & Test Metrics) |
| **10** | CO2 | L2, Dropout and Early Stopping Comparison | [`10.py`](Experiments/Exp.%2010/10.py) | `q10_plot1_stability_curves.png`, `q10_plot2_volatility_comparison.png` | Stability Assessment & Loss Comparisons |
| **11** | CO2 | CNN for Image Classification | [`11.py`](Experiments/Exp.%2011/11.py) | `q11_plot1_training_history.png`, `q11_plot2_confusion_matrix.png` | Layer Properties & Performance Metrics |
| **12** | CO2 | CNN Architecture and Hyperparameter Comparison | [`12.py`](Experiments/Exp.%2012/12.py) | `q12_plot1_architecture_comparison.png`, `q12_plot2_activation_comparison.png` | Hyperparameter & Accuracy Tables |
| **13** | CO3 | Sequential Dataset Preparation and RNN Prediction | [`13.py`](Experiments/Exp.%2013/13.py) | `q13_plot1_sine_predictions.png`, `q13_plot2_sliding_window_loss.png` | Sequence Windowing & Prediction Results |
| **14** | CO3 | Unfolded RNN Architecture Visualization | [`14.py`](Experiments/Exp.%2014/14.py) | `q14_plot1_unfolded_rnn.png`, `q14_plot2_hidden_states.png` | Unfolded Step Matrices & States |
| **15** | CO3 | BPTT Flow and Gradient Analysis | [`15.py`](Experiments/Exp.%2015/15.py) | `q15_plot1_bptt_flow.png`, `q15_plot2_vanishing_gradient.png` | Temporal Gradient Flow & Norms |
| **16** | CO3 | LSTM and RNN Performance Comparison | [`16.py`](Experiments/Exp.%2016/16.py) | `q16_plot1_rnn_vs_lstm.png`, `q16_plot2_loss_comparison.png` | Sequence Retention & Convergence Tables |
| **17** | CO3 | RNN Gradient Analysis and Gradient Clipping | [`17.py`](Experiments/Exp.%2017/17.py) | `q17_plot1_gradient_norms.png`, `q17_plot2_clipping_effect.png` | Unclipped vs Clipped Step Evaluation |
| **18** | CO3 | RNN, LSTM and GRU Hyperparameter Experimentation | [`18.py`](Experiments/Exp.%2018/18.py) | `q18_plot1_model_comparison.png`, `q18_plot2_hyperparameter_heatmaps.png` | Comparative Metric Matrices |
| **19** | CO4 | PCA and Explained Variance Analysis | [`19.py`](Experiments/Exp.%2019/19.py) | `q19_plot1_scree_plot.png`, `q19_plot2_pca_projection.png` | Eigenvalues & Variance Ratios |
| **20** | CO4 | Probabilistic PCA and Reconstruction Analysis | [`20.py`](Experiments/Exp.%2020/20.py) | `q20_plot1_reconstructions.png`, `q20_plot2_latent_space.png` | Reconstruction Errors & Noise Variances |
| **21** | CO4 | Factor Analysis and Latent Factor Interpretation | [`21.py`](Experiments/Exp.%2021/21.py) | `q21_plot1_factor_loadings.png`, `q21_plot2_latent_factors.png` | Factor Loadings & Uniqueness Tables |
| **22** | CO4 | Independent Component Analysis (ICA) | [`22.py`](Experiments/Exp.%2022/22.py) | `q22_plot1_unmixed_signals.png`, `q22_plot2_joint_distributions.png` | Mixing Matrix & Correlation Scores |
| **23** | CO4 | Bayesian Network Construction and Visualization | [`23.py`](Experiments/Exp.%2023/23.py) | `q23_plot1_network_dag.png`, `q23_plot2_posterior_marginals.png` | Conditional Probability Distributions (CPDs) |
| **24** | CO4 | Graphical Model Sampling and Probabilistic Inference | [`24.py`](Experiments/Exp.%2024/24.py) | `q24_plot1_mcmc_traces.png`, `q24_plot2_inferred_marginals.png` | MCMC Sample Frequencies & Marginals |
| **25** | CO5 | Restricted Boltzmann Machine Energy and Reconstruction | [`25.py`](Experiments/Exp.%2025/25.py) | `q25_plot1_energy_curves.png`, `q25_plot2_reconstruction_samples.png` | Free Energy Statistics & Error Rates |
| **26** | CO5 | Gibbs Sampling and Contrastive Divergence Analysis | [`26.py`](Experiments/Exp.%2026/26.py) | `q26_plot1_gibbs_chains.png`, `q26_plot2_cd_k_comparison.png` | CD-1 vs CD-5 vs CD-10 Performance |
| **27** | CO5 | Deep Belief Network Representation Learning | [`27.py`](Experiments/Exp.%2027/27.py) | `q27_plot1_layerwise_loss.png`, `q27_plot2_reconstructions.png` | Greedy Layer-Wise Training Log |
| **28** | CO5 | Deep Boltzmann Machine Simulation | [`28.py`](Experiments/Exp.%2028/28.py) | `q28_plot1_meanfield_energy.png`, `q28_plot2_reconstructions.png` | Mean-Field Inference Logs |
| **29** | CO5 | VAE / GAN Sample Generation | [`29.py`](Experiments/Exp.%2029/29.py) | `q29_plot1_vae_samples.png`, `q29_plot2_gan_training_curves.png` | Sample Evaluation & Loss Profiles |
| **30** | CO5 | Generative Model Output Comparison and Evaluation | [`30.py`](Experiments/Exp.%2030/30.py) | `q30_plot1_fidelity_scores.png`, `q30_plot2_sample_diversity.png` | Comparative Metric Evaluation Table |

---

## 🛠️ Requirements & Installation

Clone the repository and install the standard dependencies:

```bash
git clone https://github.com/Tamilarasan13092005/Deep-learning.git
cd Deep-learning

pip install numpy pandas matplotlib scipy scikit-learn
pip install tensorflow keras networkx pgmpy
```

---

## 🚀 Running an Experiment

Navigate into any experiment folder and run its script:

```bash
cd "Experiments/Exp. 1"
python p1.py
```

All plots (`*.png`) and structured output tables (`*.png` / `*.csv`) will be displayed and saved in the corresponding experiment directory.

---

## 👤 Author

**Tamilarasan M**  
GitHub: [@Tamilarasan13092005](https://github.com/Tamilarasan13092005)
