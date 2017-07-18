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

"""input represents how many dimensions the input has (age and size)"""
"""label represents the number of output classes, which is 2 in this case (either malignant or benign"""
input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

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

### Training

# Using cross-entropy to measure the loss from the ground truth
label = C.input_variable((num_output_classes), np.float32)
loss = C.cross_entropy_with_softmax(z, label)

eval_error = C.classification_error(z, label)

# Configure training
"""Stochastic gradient descent - Trains over different samples over time to minimize the losses"""
"""The learning rate is how much we change the parameters in any iteration"""

# Instantiate the trainer object to drive the model training
learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

# Defines a utility function to compute the moving average sum
"""A more efficient implementation is possible with np.cumsum() function"""


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error


# Initializing parameters for the trainer
minibatch_size = 25
num_samples = 20000
num_minibatches_to_train = num_samples / minibatch_size

# Running the trainer
training_progress_output_freq = 20

plotdata = {"batchsize": [], "loss": [], "error": []}

for i in range(0, int(num_minibatches_to_train)):
    features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)

    # Specify the input variables mapping in the model to actual minibatch data for training
    trainer.train_minibatch({input: features, label: labels})
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=0)

    if not (loss == "NA" or error == "NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

# Compute the moving average loss to smooth out the noise in SGD
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error
import matplotlib.pyplot as plt

plt.figure(1)

plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label prediction error')
plt.title('Minibatch run vs Label prediction error')

plt.show()

### Evaluation

# Generate new data
test_minibatch_size = 25
features, labels = generate_random_data_sample(test_minibatch_size, input_dim, num_output_classes)

trainer.test_minibatch({input: features, label: labels})

# Route the network output via softmax
out = C.softmax(z)

predicted_label_probs = out.eval({input: features})

print("Label    :", [np.argmax(label) for label in labels])
print("Predicted:", [np.argmax(row) for row in predicted_label_probs])
