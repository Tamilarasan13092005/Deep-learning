# Deep Learning Experiments

A comprehensive collection of **30 Python experiments** covering foundational to advanced deep learning topics, organized by Course Outcomes (CO).

---

## 📁 Repository Structure

```
Deep-learning/
├── Experiment_01_Vector_Matrix_Operations/
├── Experiment_02_Covariance_Eigenvalues/
├── Experiment_03_Probability_Distributions/
├── Experiment_04_Entropy_CrossEntropy_KL/
├── Experiment_05_Bias_Variance_Overfitting/
├── Experiment_06_Gradient_Descent_Optimizers/
├── Experiment_07_Perceptron_Forward_Propagation/
├── Experiment_08_Two_Layer_NN_Backpropagation/
├── Experiment_09_MLP_Implementation/
├── Experiment_10_L2_Dropout_EarlyStopping/
├── Experiment_11_CNN_Image_Classification/
├── Experiment_12_CNN_Architecture_Comparison/
├── Experiment_13_RNN_Sequential_Prediction/
├── Experiment_14_RNN_Architecture_Visualization/
├── Experiment_15_BPTT_Gradient_Analysis/
├── Experiment_16_LSTM_RNN_Comparison/
├── Experiment_17_RNN_Gradient_Clipping/
├── Experiment_18_RNN_LSTM_GRU_Hyperparameter/
├── Experiment_19_PCA_Explained_Variance/
├── Experiment_20_Probabilistic_PCA/
├── Experiment_21_Factor_Analysis/
├── Experiment_22_ICA_Component_Visualization/
├── Experiment_23_Bayesian_Network/
├── Experiment_24_Graphical_Model_Sampling/
├── Experiment_25_RBM_Energy_Reconstruction/
├── Experiment_26_Gibbs_Contrastive_Divergence/
├── Experiment_27_Deep_Belief_Network/
├── Experiment_28_DBM_ConvBM_Simulation/
├── Experiment_29_VAE_GAN_Generation/
└── Experiment_30_Generative_Model_Comparison/
```

---

## 📋 Experiment List

| S.No | CO  | Experiment | Packages |
|------|-----|------------|----------|
| 1  | CO1 | Vector and Matrix Operations for Deep Learning | NumPy, Matplotlib |
| 2  | CO1 | Covariance, Eigenvalues and Eigenvectors Analysis | NumPy, Pandas, Matplotlib |
| 3  | CO1 | Probability Distributions and Bayes' Theorem | NumPy, SciPy, Matplotlib |
| 4  | CO1 | Entropy, Cross-Entropy and KL Divergence | NumPy, SciPy, Matplotlib |
| 5  | CO1 | Bias–Variance, Overfitting and Underfitting Analysis | Scikit-learn, NumPy, Matplotlib |
| 6  | CO1 | Gradient Descent, SGD, Momentum and Adam Comparison | NumPy, TensorFlow, Keras, Matplotlib |
| 7  | CO2 | Perceptron and Forward Propagation | NumPy, Matplotlib |
| 8  | CO2 | Two-Layer Neural Network with Backpropagation | NumPy, Matplotlib |
| 9  | CO2 | Multilayer Perceptron Implementation | TensorFlow, Keras, NumPy, Matplotlib |
| 10 | CO2 | L2, Dropout and Early Stopping Comparison | TensorFlow, Keras, NumPy, Matplotlib |
| 11 | CO2 | CNN for Image Classification | TensorFlow, Keras, NumPy, Matplotlib |
| 12 | CO2 | CNN Architecture and Hyperparameter Comparison | TensorFlow, Keras, NumPy, Matplotlib |
| 13 | CO3 | Sequential Dataset Preparation and RNN Prediction | TensorFlow, Keras, NumPy, Pandas, Matplotlib |
| 14 | CO3 | Unfolded RNN Architecture Visualization | TensorFlow, Keras, NetworkX, Matplotlib |
| 15 | CO3 | BPTT Flow and Gradient Analysis | TensorFlow, Keras, NumPy, Matplotlib |
| 16 | CO3 | LSTM and RNN Performance Comparison | TensorFlow, Keras, NumPy, Pandas, Matplotlib |
| 17 | CO3 | RNN Gradient Analysis and Gradient Clipping | TensorFlow, Keras, NumPy, Matplotlib |
| 18 | CO3 | RNN, LSTM and GRU Hyperparameter Experimentation | TensorFlow, Keras, NumPy, KerasTuner, Matplotlib |
| 19 | CO4 | PCA and Explained Variance Analysis | NumPy, Scikit-learn, Matplotlib |
| 20 | CO4 | Probabilistic PCA and Reconstruction Analysis | NumPy, Scikit-learn, Matplotlib |
| 21 | CO4 | Factor Analysis and Latent Factor Interpretation | Scikit-learn, NumPy, Pandas, Matplotlib |
| 22 | CO4 | Independent Component Analysis and Component Visualization | Scikit-learn, NumPy, Matplotlib |
| 23 | CO4 | Bayesian Network Construction and Visualization | pgmpy, NetworkX, Matplotlib |
| 24 | CO4 | Graphical Model Sampling and Probabilistic Inference | pgmpy, NumPy, Pandas, NetworkX, Matplotlib |
| 25 | CO5 | Restricted Boltzmann Machine Energy and Reconstruction | NumPy, SciPy, Matplotlib |
| 26 | CO5 | Gibbs Sampling and Contrastive Divergence Analysis | NumPy, SciPy, Matplotlib |
| 27 | CO5 | Deep Belief Network Representation Learning | NumPy, SciPy, TensorFlow, Keras, Matplotlib |
| 28 | CO5 | Deep Boltzmann Machine / Convolutional Boltzmann Machine Simulation | NumPy, SciPy, TensorFlow, Keras, Matplotlib |
| 29 | CO5 | VAE/GAN Sample Generation | TensorFlow, Keras, NumPy, Matplotlib |
| 30 | CO5 | Generative Model Output Comparison and Evaluation | TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Tamilarasan13092005/Deep-learning.git
cd Deep-learning

# Install dependencies
pip install numpy pandas matplotlib scipy scikit-learn
pip install tensorflow keras
pip install networkx pgmpy keras-tuner
```

---

## ▶️ Running an Experiment

```bash
cd Experiment_01_Vector_Matrix_Operations
python experiment01.py
```

Each experiment:
- Prints results to the console
- Saves output plot(s) as `output_expXX.png` in its folder

---

## 📦 Package Requirements

```
numpy
pandas
matplotlib
scipy
scikit-learn
tensorflow>=2.0
keras
networkx
pgmpy
keras-tuner
```

> **Note:** Experiments 1–8, 14–15, 19–26, 25–26 work with **NumPy/SciPy only** (no TensorFlow required).  
> Experiments requiring TensorFlow include a NumPy fallback where possible.

---

## 👨‍💻 Author

**Tamilarasan** | [GitHub](https://github.com/Tamilarasan13092005)
