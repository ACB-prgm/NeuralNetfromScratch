import numpy as np


class Loss:
    def __init__(self, func="categorical_cross_entropy") -> None:
        self.predicted = None
        self.trues = None
        self.func = func
        self.batch_loss = 0.0
        self.der_loss = None
        self.gradients = None
    
    def calc_loss(self, model_output, trues, layers):
        self.predicted = model_output
        self.trues = trues

        self.batch_loss = np.mean(getattr(self, self.func)()) # averages the sample losses to represent the batch loss
        
        for layer in layers:
            self.batch_loss += layer.reg_loss

        self.gradients = getattr(self, f"der_{self.func}")() # starts back propagation

    def categorical_cross_entropy(self):
        # used when there are multiple classes (categories) where one class is the correct answer. ex: a cat class and a dog class
        pred_clipped = np.clip(self.predicted, 1e-7, 1 - 1e-7) # ensures there are no values == 0 which would result in inf when log() used
        
        # get relevant pred values.  IE those that correspond to our desired trues
        if len(self.trues.shape) == 1: # if true given in a one dimensional array where idx = row and value = column (one-hot encoded)
            # pred = [[.9, .1], [.7, .3], [.2, .8]], true = [0, 0, 1]
            relevants = pred_clipped[range(len(self.trues)), self.trues]
        elif len(self.trues.shape) == 2: # if true given in an explicit 2-dimensional array
            # pred = [[.9, .1], [.7, .3], [.2, .8]], true = [[1, 0], [1, 0], [0, 1]]
            relevants = np.sum(pred_clipped * self.trues, axis=1)
        
        return(-np.log(relevants))
    
    def der_categorical_cross_entropy(self):
        # the derivative of CCE is simply -1 * true/pred
        # If labels are sparse, turn them into one-hot vector
        if len(self.trues.shape) == 1:
            self.trues = np.eye(len(self.predicted[0]))[self.trues]

        return (-self.trues / self.predicted) / len(self.trues) # we div by # samples to normalize the gradient for the optimizer

    def binary_cross_entropy(self):
        # self.trues must be a list of lists of binary values, each value corresponding to a respective neuron. eg [[0], [1]]

        # used when each neuron represents two classes, yes/no typically. One neuron = cat or not cat, but could also represent cat or dog
        pred_clipped = np.clip(self.predicted, 1e-7, 1 - 1e-7) # ensures there are no values == 0 which would result in inf when log() used
        
        
        if len(self.trues.shape) == 1:
            self.trues = np.eye(len(self.predicted[0]))[self.trues]

        sample_losses = -(self.trues * np.log(pred_clipped) + (1 - self.trues) * np.log(1 - pred_clipped)) # -1 * ( (true * log(pred)) + ((1-true) * log(1-pred)) )

        return np.mean(sample_losses, axis=-1)

    def der_binary_cross_entropy(self):
        pred_clipped = np.clip(self.predicted, 1e-7, 1 - 1e-7) # ensures there are no values == 0 which would result in inf when log() used
        gradients = -(self.trues / pred_clipped - (1 - self.trues) / (1 - pred_clipped)) / len(self.predicted[0]) 

        return gradients / len(self.predicted)
    
    def mean_squared_error(self):
        # (∑ (true - pred)**2) / num_outputs  ie == the mean of the squared differences of true and predicted values
        return np.mean((self.trues - self.predicted)**2, axis=-1)

    
    def der_mean_squared_error(self):
        num_samples = len(self.predicted)
        num_outputs = len(self.predicted[0])

        gradients = -2 * (self.trues - self.predicted) / num_outputs # ∂/∂x of (∑ (true - pred)**2) / num_outputs = -2 * (true-predicted) / num_outputs

        return gradients / num_samples # normalize by number of samples
    
    def mean_absolute_error(self):
        # like MSE, but less aggressive because it is linear rather than exponential.  
        # It is used less frequently than MSE, but can also be paired with linear activation functions for scalar output
        # (∑ |true - pred|) / num_outputs  ie == the mean of the abs(differences) of true and predicted values
        return np.mean(np.abs(self.trues - self.predicted), axis=-1)
    
    def der_mean_absolute_error(self):
        num_samples = len(self.predicted)
        num_outputs = len(self.predicted[0])

        gradients = -np.sign(self.trues - self.predicted) / num_outputs # ∂/∂x = -1/num_outputs * (1 if diff>0 and -1 if diff<0) (bc derivative of abs val)

        return gradients / num_samples