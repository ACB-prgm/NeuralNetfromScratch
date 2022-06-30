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

        self.batch_loss = np.mean(getattr(self, self.func)()) # averages the losses to represent the batch loss
        for layer in layers:
            self.batch_loss += layer.reg_loss

        self.gradients = getattr(self, f"der_{self.func}")() # starts back propagation

    def categorical_cross_entropy(self):
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
        pass