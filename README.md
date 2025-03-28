# A Deep Learning Prediction Model for On-Site Earthquake Early Warning in India

![image](https://github.com/PavanMohanN/EEW_system_Variational/assets/65588614/7673bf77-604a-4b8a-9bcd-53fada23d96c)

## Overview

This repository contains code for an earthquake early warning system based on a variational autoencoder (VAE). The system is designed to predict 27 spectral accelerations using 8 input variables: Peak Ground Acceleration ($PGA$), Peak Ground Displacement ($PGD$), Predominant Frequency ($F_p$), 5-95% Significant Duration  ($T_{sig}$), Arias Intensity ($I_a$), Cumulative Absolute Velocity  ($CAV$), Site Class ($S_c$), and direction flag ($dir$). 

## Library Installation and Importing Libraries

### Installation

Ensure you have Python 3.x installed. Use the following pip commands to install the necessary libraries:

<code>pip install numpy pandas matplotlib scikit-learn keras tensorflow</code><br>

<h3> Importing the libraries </h3>

<code>import numpy as np</code><br>
<code>import pandas as pd</code><br>
<code>import matplotlib.pyplot as plt</code><br>
<code>from sklearn.model_selection import train_test_split</code><br>
<code>from sklearn.preprocessing import MinMaxScaler, StandardScaler</code><br>
<code>from scipy.stats import kurtosis</code><br>
<code>from keras.layers import Input, Dense, Lambda</code><br>
<code>from keras.models import Model, Sequential, load_model</code><br>
<code>from keras import backend as K</code><br>
<code>from keras.losses import mse</code><br>
<code>from keras.optimizers import Adam</code><br>
<code>from keras.callbacks import ModelCheckpoint, Callback</code><br>
<code>import tensorflow as tf</code><br>

## About the libraries

- numpy: For numerical operations and array manipulations.
- pandas: For data manipulation and handling.
- matplotlib.pyplot: For plotting graphs and visualizations.
- sklearn.model_selection.train_test_split: For splitting data into training and testing sets.
- sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.StandardScaler: For scaling numerical input data.
- keras.layers: For defining layers in the neural network model.
- keras.models: For defining and manipulating the neural network models.
- keras.backend: Provides operations that are not yet part of the official Keras API.
- keras.losses.mse: Mean squared error loss function, commonly used in regression tasks.
- keras.optimizers.Adam: Optimizer algorithm for gradient-based optimization.
- tensorflow: Backend framework for Keras deep learning library.
- keras.callbacks.ModelCheckpoint: Callback to save the model after every epoch.
- keras.callbacks.Callback: Base class for Keras callbacks.

## Architecture

The core of the system is a VAE, which is trained to map spectral accelerations to themselves. The VAE consists of:
- *Architecture*: CVAE includes an encoder, a mapping layer, and a decoder.
- *Encoder*: Reduces $S_a(T)$ to a latent space.
- *Mapping Layers*: Incorporates 8 conditional inputs (GMPs) into the latent space.
- *Decoder*: Reconstructs $S_a(T)$ from the latent space.

### Hyperparameters

- *Network Structure*: Symmetric encoder and decoder with hidden layers.
- *Latent Space*: Dimensionality of 3 ($z_1, z_2, z_3$).
- *Mapping Layers*: Two hidden layers with 4 and 3 nodes, using ReLU activation.

### Loss Function

- *Objective*: Minimize combined reconstruction ( $L_{recon}$ ) and regularization ( $L_{reg}$ ) losses.
- $L_{recon}$: Measures decoder's ability to reconstruct Sa(T).
- $L_{reg}$: Encourages latent space to approximate a standard Gaussian distribution.

### Model Training

- *Training Strategy*: K-fold Cross Validation, Adam optimizer.
- *Initial Training*: VAE trained for 50 epochs (batch size: 16).
- *Mapping Network*: Trained for 600 epochs (batch size: 64) to optimize conditional input mapping.

![image](https://github.com/PavanMohanN/EEW_system_Variational/assets/65588614/0cb249d7-d8ba-4903-9195-d13aa7cce51a)


Fig. 1. Depiction of True Vs Recorded Values of Spectral Acceleration.

### Sensitivity Study

- *Analysis*: Evaluates model response to conditional inputs (GMPs).
- *Visualization*: Latent variables ($z_1, z_2, z_3$) (herein the code the following variables were named LA1, LA2 and LA3 respectively) mapped against conditional inputs to validate model effectiveness.

- The *Mapping2Output.ipynb* file explains the procedure till here.

![image](https://github.com/PavanMohanN/EEW_system_Variational/assets/65588614/9f1ca449-893b-4a17-a289-038fb3b17f9f)

Fig. 2. Sensitivity with respect to various parameters.

## Input Mapping Layers - The game changer
Given that spectral accelerations are not known beforehand in an early warning scenario, additional layers are designed to map the input variables directly to the latent space of the VAE. This architecture involves:
- *Two separate layers*: These map the 8 input variables ($PGA, PGD, F_p, T_{sig}, {I_a}, CAV, S_c, dir$) to the latent space of the VAE.
- *Concatenation*: The encoder component of the VAE is disconnected, and the layers mapping inputs to the latent space are concatenated directly.

- For concatenation of mapping layers with the decoder, and creating a standaone network, refer to *Concatenation2Compact.ipynb*

## Output

This architecture facilitates a direct mapping from the input variables to the predicted spectral accelerations, enabling real-time earthquake early warning predictions.

`Created in Jul 2024`

`File: Mapping2Output.ipynb`

`@author: Pavan Mohan Neelamraju`

`Affiliation: Indian Institute of Technology Madras`

**Email**: npavanmohan3@gmail.com

**Personal Website 🔴🔵**: [[pavanmohan.netlify.app](https://pavanmohan.netlify.app/)]
