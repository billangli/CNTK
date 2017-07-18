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
