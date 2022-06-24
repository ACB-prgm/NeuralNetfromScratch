import numpy as np


class LayerDense:
    def __init__(self, num_inputs, num_neurons) -> None:
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = 0.1 * np.random.standard_normal((num_inputs, num_neurons)) # creates weights for each neuron.  one weight for each input / input neuron
        self.biases = np.zeros((1, num_neurons)) # creates array of np.float zeros with length = num_neurons in layer
        self.outputs = []
    
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def apply_ReLU(self):
        # applies the ReLU activation function to all raw outputs.
        self.outputs = np.maximum(0, self.outputs)
    
    def apply_softmax(self):
        # applies the softmax function to all raw outputs
        # this is done on the final output layer to prepare for back propagation
        exps = np.exp(self.outputs - np.max(self.outputs, axis=1, keepdims=True))
        self.outputs = exps / np.sum(exps, axis=1, keepdims=True)