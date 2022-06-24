import numpy as np


class Loss:
    def __init__(self, model_output, true) -> None:
        self.predicted = model_output
        self.true = true
        self.batch_loss = 0.0
    
    def calc_loss(self, func="categorical_cross_entropy"):
        sample_losses = getattr(Loss, func)(self)
        self.batch_loss = np.mean(sample_losses)

    def categorical_cross_entropy(self):
        pred_clipped = np.clip(self.predicted, 1e-7, 1 - 1e-7) # ensures there are no values == 0 which would result in inf when log() used

        if len(self.true.shape) == 1: # if true given in a one dimensional array wher idx = row and value = column
            confidences = pred_clipped[range(len(pred_clipped)), self.true]
        elif len(self.true.shape) == 2: # if true given in 2-dimensional one-hot encoded array
            confidences = np.sum(pred_clipped * self.true, axis=1)
        
        return(-np.log(confidences))