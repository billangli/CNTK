"""
FeedforwardNetwork.py
Microsoft CNTK Tutorial 102
Classifying simulated cancer data using a feedforward network
Bill Li
Jul. 18th, 2017
"""

# Import the relevant components
from __future__ import print_function

import os

import cntk as C
from cntk import *

# Select the right target device when this notebook is being tested
if 'test device' in os.environ:
    if os.environ['test device'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

### Data Generation

# Define the network
"""We have two choices of input: age & tumour size,
as for output, it will be either malignant or benign, represented by 0 & 1"""
input_dim = 2
num_output_classes = 2

"""We are generating synthetic data in this tutorial using the numpy library"""

# Ensure we always get the same amount of randomness
"""the seed being 0 makes the random numbers predictable because each seed is reset"""
np.random.seed(0)


# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)

    # Specify the data type to match the input variable used later
    # (default type is double)
    X = X.astype(np.float32)

    # Converting class 0 into the vector "1 0 0", class 1 into "0 1 0"
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


# Create the input variables denoting the features and the label data. Note: the input
# does not need additional info on number of observations (Samples) since CNTK creates only
# the network topology first
mysamplesize = 32
features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)

# Plot the data
import matplotlib.pyplot as plt

# Given this is a 2 class ()
colors = ['r' if l == 0 else 'b' for l in labels[:, 0]]

plt.scatter(features[:, 0], features[:, 1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()

### Model Creation
"""We will have 2 hidden layers each with 50 nodes (dimension)"""

num_hidden_layers = 2
hidden_layers_dim = 50

# Defining a linear layer
"""evidence(z) = weight(W) x feature(x) + bias term(b)"""


def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]

    weight = C.parameter(shape=(input_dim, output_dim))
    bias = C.parameter(shape=(output_dim))

    return bias + C.times(input_var, weight)


# Defining one hidden layer
def dense_layer(input_var, output_dim, nonlinearity):
    l = linear_layer(input_var, output_dim)

    return nonlinearity(l)


# Defining a multilayer feedforward classification model
def fully_connected_classifier_net(input_var, num_output_classes, hidden_layers_dim, num_hidden_layers, nonlinearity):
    h = dense_layer(input_var, hidden_layers_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layers_dim, nonlinearity)

    return linear_layer(h, num_output_classes)


# Creating the fully connected classifier
z = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, num_hidden_layers, C.sigmoid)
