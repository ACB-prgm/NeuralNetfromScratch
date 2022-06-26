import numpy as np


class Loss:
    def __init__(self, model_output, true) -> None:
        self.predicted = model_output
        self.true = true
        self.batch_loss = 0.0
    
    def calc_loss(self, func="categorical_cross_entropy"):
        sample_losses = getattr(Loss, func)(self)
        self.batch_loss = np.mean(sample_losses) # averages the losses to represent the batch loss
        # this is done to save on computing power and time

    def categorical_cross_entropy(self):
        pred_clipped = np.clip(self.predicted, 1e-7, 1 - 1e-7) # ensures there are no values == 0 which would result in inf when log() used
        
        # get confidences
        if len(self.true.shape) == 1: # if true given in a one dimensional array where idx = row and value = column
            # pred = [[.9, .1], [.7, .3], [.2, .8]], true = [0, 0, 1]
            confidences = pred_clipped[range(len(self.true)), self.true]
        elif len(self.true.shape) == 2: # if true given in an explicit 2-dimensional array
            # pred = [[.9, .1], [.7, .3], [.2, .8]], true = [[1, 0], [1, 0], [0, 1]]
            confidences = np.sum(pred_clipped * self.true, axis=1)
        
        return(-np.log(confidences))