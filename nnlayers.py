import numpy as np


class LayerDense:
    def __init__(self, num_inputs, num_neurons, activation="ReLU", dropout=0.0,
                wt_reg_l1=0.0, wt_reg_l2=0.0, bs_reg_l1=0.0, bs_reg_l2=0.0) -> None:
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = 0.01 * np.random.standard_normal((num_inputs, num_neurons)) # creates weights for each neuron.  one weight for each input / input neuron
        self.biases = np.zeros((1, num_neurons)) # creates array of np.float zeros with length = num_neurons in layer
        self.inputs = None
        self.outputs = None
        self.raw_outputs = None
        self.der_weights = None 
        self.der_biases = None 
        self.der_inputs = None

        self.binary_mask = None
        self.dropout = 1 - dropout
        self.wt_reg_l1 = wt_reg_l1
        self.wt_reg_l2 = wt_reg_l2
        self.bs_reg_l1 = bs_reg_l1
        self.bs_reg_l2 = bs_reg_l2
        self.reg_loss = 0

    def forward(self, inputs, training=False):
        self.inputs = inputs # inputs from previous layer or sample inputs
        self.raw_outputs = np.dot(inputs, self.weights) + self.biases # (w0*i0 ... wnin) + bias
        self.outputs = getattr(self, self.activation)(self.raw_outputs) # activated outputs

        if training and self.dropout:
            self.binary_mask = np.random.binomial(1, self.dropout, size=self.outputs.shape) / self.dropout
            self.outputs *= self.binary_mask

        self.reg_loss = self.regularization_loss()

    def backwards(self, gradients, training=False):
        if training and self.dropout:
            gradients *= self.binary_mask
          
        gradients = getattr(self, f"der_{self.activation}")(gradients) # multiplies the gradients from the previous layer by the derivative of the activation function.

        self.der_biases = np.sum(gradients, axis=0, keepdims=True) # == gradient (bc d/dx of biases is always 1.0 bc it is a sum)
        self.der_weights = np.dot(self.inputs.T, gradients) # == inputs * gradient (bc d/dw of w*i = i)

        self.der_reg_loss()

        self.der_inputs = np.dot(gradients, self.weights.T) # == weights * gradient (bc d/di of w*i = w)
        self.gradients = self.der_inputs

    def regularization_loss(self):
        # calculates the regularization loss for a given layer.
        # these are summed with the batch loss as a "penalty" for large weights and biases
        # in order to prevent overfitting / reduce generalization error.
        # L1 (linear) penalizes all weights/biases evenly, and thus can cause some to approach 0 and be muted 
        #   - used for features selection (reduction of "unecessary" neurons)
        # L2 (exp) penalizes larger weights/biases more than small ones, 
        #   - used to increase accuracy in complex models by dispersing
        reg_loss = 0

        if self.wt_reg_l1:
            reg_loss += self.wt_reg_l1 * np.sum(np.abs(self.weights))
        if self.wt_reg_l2:
            reg_loss += self.wt_reg_l2 * np.sum(self.weights * self.weights)
        if self.bs_reg_l1:
            reg_loss += self.bs_reg_l1 * np.sum(np.abs(self.biases))
        if self.bs_reg_l2:
            reg_loss += self.bs_reg_l2 * np.sum(self.biases * self.biases)

        return reg_loss

    def der_reg_loss(self):
        # Gradients on regularization
        if self.wt_reg_l1: # l1' = w>0 -> 1 || w<0 -> -1 
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.der_weights += self.wt_reg_l1 * dL1
        if self.wt_reg_l2: # l2' = 2 * λ * weights
            self.der_weights += 2 * self.wt_reg_l2 * self.weights
        if self.bs_reg_l1:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.der_biases += self.bs_reg_l1 * dL1
        if self.bs_reg_l2:
            self.der_biases += 2 * self.bs_reg_l2 * self.biases


    def ReLU(self, raw_outputs):
        # applies the ReLU activation function to all raw outputs.
        # used for hidden layers
        return np.maximum(0, raw_outputs)
    
    def der_ReLU(self, gradients):
        der_ReLU = gradients.copy()
        der_ReLU[self.raw_outputs <= 0] = 0
        
        return der_ReLU
    
    def linear(self, raw_ouputs):
        # Used on the final layer when targets are scalar values
        # it is paired with Mean Squared Error Loss or Mean Absolute Error
        return raw_ouputs # bc linear is y=x
    
    def der_linear(self, gradients): # derivative of y=x is 1
        return gradients.copy()
    
    def softmax(self, raw_outputs): 
        # applies the softmax function to all raw outputs
        # Function: https://en.wikipedia.org/wiki/Softmax_function
        # softmax is used on the final output layer when the there is one correct category/class
        # it is paired with the Categorical Cross Entropy loss calculation
        
        exps = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True)) # eulers number (e) to outputh power to remove negatives while maintaining meaning
        # the max is removed from each to prevent exponential explosion and keep numbers low/appropriate
        return exps / np.sum(exps, axis=1, keepdims=True) # normalizes into probability distribution for loss calculation
    
    def der_softmax(self, gradients):
        # You can actually combine the derivatives of the softmax and CCE loss to gain a 6-7x speed boost, but I will leave
        # this as-is as this is just a learning exercize and I would have to break my class system.
        der_softmax = np.empty_like(gradients)

        # iterate sample-wise over pairs of the outputs and gradients, calculating the partial derivatives and applying the chain rule
        for index, (single_output, single_gradients) in enumerate(zip(self.outputs, gradients)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            der_softmax[index] = np.dot(jacobian_matrix, single_gradients)
        
        return der_softmax

    def sigmoid(self, raw_outputs): # applies the signmoid function to all raw outputs
        # Function: https://en.wikipedia.org/wiki/Sigmoid_function
        # it is paired with the Binary Cross Entropy loss calculation

        return 1 / (1 + np.exp(-raw_outputs))
    
    def der_sigmoid(self, gradients):
        return gradients * self.outputs * (1 - self.outputs) # d/dsig = sig * (1 - sig)