import numpy as np


class LayerDense:
    def __init__(self, num_inputs, num_neurons, activation="ReLU") -> None:
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = 0.1 * np.random.standard_normal((num_inputs, num_neurons)) # creates weights for each neuron.  one weight for each input / input neuron
        self.biases = np.zeros((1, num_neurons)) # creates array of np.float zeros with length = num_neurons in layer
        self.inputs = None
        self.raw_outputs = None
        self.outputs = None

        self.der_weights = None 
        self.der_biases = None 
        self.der_inputs = None 
        self.gradients = None

    def apply_ReLU(self, raw_outputs):
        # applies the ReLU activation function to all raw outputs.
        self.raw_outputs = raw_outputs
        return np.maximum(0, raw_outputs)
    
    def apply_der_ReLU(self, gradients):
        der_ReLU = gradients.copy()
        der_ReLU[self.raw_outputs <= 0] = 0
        return der_ReLU
    
    def apply_softmax(self, raw_outputs):
        # applies the softmax function to all raw outputs
        # this is done on the final output layer to prepare for back propagation
        exps = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True)) # eulers number (e) to outputh power to remove negatives while maintaining meaning
        # the max is removed from each to prevent exponential explosion and keep numbers low/appropriate
        return exps / np.sum(exps, axis=1, keepdims=True) # normalization, why?
    
    def forward(self, inputs):
        self.inputs = inputs # inputs from previous layer or sample inputs
        raw_outputs = np.dot(inputs, self.weights) + self.biases # (w0*i0 ... wnin) + bias
        self.outputs = getattr(self, f"apply_{self.activation}")(raw_outputs) # activated outputs

    def backwards(self, gradients):
        gradients = getattr(self, f"apply_der_{self.activation}")(self, gradients) # takes the gradients from the previous layer 
        # and multiplies them by the derivative of the activation function.

        self.der_biases = np.sum(gradients, axis=0, keepdims=True) # == d/dx of biases * the gradient (ie == the gradient bc d/dx of a sum is always 1.0)
        self.der_weights = np.dot(self.inputs.T, gradients) # == the inputs (bc d/dw of w*i = i)
        self.der_inputs = np.dot(gradients, self.weights.T) # == the weights (bc d/di of w*i = w)
        self.gradients = np.dot(self.der_inputs, gradients)