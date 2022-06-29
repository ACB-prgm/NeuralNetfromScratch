import numpy as np


class LayerDense:
    def __init__(self, num_inputs, num_neurons, activation="ReLU") -> None:
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = 0.1 * np.random.standard_normal((num_inputs, num_neurons)) # creates weights for each neuron.  one weight for each input / input neuron
        self.biases = np.zeros((1, num_neurons)) # creates array of np.float zeros with length = num_neurons in layer
        self.inputs = None
        self.outputs = None
        self.raw_outputs = None

        self.der_weights = None 
        self.der_biases = None 
        self.der_inputs = None 

    def forward(self, inputs):
        self.inputs = inputs # inputs from previous layer or sample inputs
        raw_outputs = np.dot(inputs, self.weights) + self.biases # (w0*i0 ... wnin) + bias
        self.outputs = getattr(self, self.activation)(raw_outputs) # activated outputs

    def backwards(self, gradients):        
        gradients = getattr(self, f"der_{self.activation}")(gradients) # takes the gradients from the previous layer 
        # and multiplies them by the derivative of the activation function.

        self.der_biases = np.sum(gradients, axis=0, keepdims=True) # == gradient (bc d/dx of biases is always 1.0 bc it is a sum)
        self.der_weights = np.dot(self.inputs.T, gradients) # == inputs * gradient (bc d/dw of w*i = i)
        self.der_inputs = np.dot(gradients, self.weights.T) # == weights * gradient (bc d/di of w*i = w)
        self.gradients = self.der_inputs

    def ReLU(self, raw_outputs):
        # applies the ReLU activation function to all raw outputs.
        self.raw_outputs = raw_outputs
        return np.maximum(0, raw_outputs)
    
    def der_ReLU(self, gradients):
        der_ReLU = gradients.copy()
        der_ReLU[self.raw_outputs <= 0] = 0
        return der_ReLU
    
    def softmax(self, raw_outputs): # applies the softmax function to all raw outputs
        # Function: https://en.wikipedia.org/wiki/Softmax_function
        # softmax is used on the final output layer when the target has >2 classes
        # it is paired with the Categorical Cross Entropy loss calculation
        
        exps = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True)) # eulers number (e) to outputh power to remove negatives while maintaining meaning
        # the max is removed from each to prevent exponential explosion and keep numbers low/appropriate
        return exps / np.sum(exps, axis=1, keepdims=True) # normalizes into probability distribution for loss calculation
    
    def der_softmax(self, gradients):
        # You can actually combine the derivatives of the softmax and CCE loss to gain a 6-7x speed boost, but I will leave
        # this as-is as this is just a learning exersize and I would have to break my class system.
        der_softmax = np.empty_like(gradients)

        # iterate sample-wise over pairs of the outputs and gradients, calculating the partial derivatives and applying the chain rule
        for index, (single_output, single_gradients) in enumerate(zip(self.outputs, gradients)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            der_softmax[index] = np.dot(jacobian_matrix, single_gradients)
        
        return der_softmax

    def sigmoid(self, raw_outputs): # applies the signmoid function to all raw outputs
        # Function: https://en.wikipedia.org/wiki/Sigmoid_function
        # sigmoid is used on the final output layer when the target has <=2 classes
        # it is paired with the Binary Cross Entropy loss calculation
        return 1 / (1 + np.exp(-1 * raw_outputs))