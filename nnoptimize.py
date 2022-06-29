import numpy as np


class StochasticGradientDecent:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0) -> None:
        self.learning_rate = learning_rate # essentially the step size for gradient decent
        self.current_lr = learning_rate
        self.decay = decay # the amount / rate of decay of the lr so that steps become smaller near global minima
        self.momentum = momentum # applies history of directionality to steps to prevent becoming stuck in local minima
        self.iterations = 0
    
    def pre_update(self):
        # update current learning rate by decay/iterations
        if self.decay:
            self.current_lr = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'wt_momentums'): # create empty past momentum arrays if they dont already exist
                layer.wt_momentums = np.zeros_like(layer.weights)
                layer.bs_momentums = np.zeros_like(layer.biases)
            
            # take previous increments multiplied by momentum and update with current gradients
            weight_increments = self.current_lr * layer.der_weights + self.momentum * layer.wt_momentums # increment(m) = (learning_rate * der_param) + (momentum * increment(m-1))
            bias_increments = self.current_lr * layer.der_biases + self.momentum * layer.bs_momentums # ^ this is just the normal increment + momentum
            layer.wt_momentums = weight_increments # cache the momentum in the layer for use in next iteration
            layer.bs_momentums = bias_increments
        else: # normal SGD increment
            weight_increments = self.current_lr * layer.der_weights # learning_rate * gradient
            bias_increments = self.current_lr * layer.der_biases

        # increment weights and biases
        layer.weights -= weight_increments
        layer.biases -= bias_increments
    
    def post_update(self):
        # after update
        self.iterations += 1
    

class AdamAdaptiveMomentum:
    # similar to SGD w/ momentum, but with an lr adapted for each individual perameter
    # this makes changes in learning rate smoother individually and more cohesive globally
    # also utilizes beta to "warm up" faster with the initial steps
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.eps = epsilon # prevents division by zero error
        self.beta_1 = beta_1 # corrects for stwarting with all zeros. cache (b2) and momentum (b1) are divideed
        self.beta_2 = beta_2 # by 1-beta as beta approaches 0 w/ each iter

    def pre_update(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'wt_cache'): # create empty cache arrays if they dont already exist
            layer.wt_momentums = np.zeros_like(layer.weights)
            layer.wt_cache = np.zeros_like(layer.weights)
            layer.bs_momentums = np.zeros_like(layer.biases)
            layer.bs_cache = np.zeros_like(layer.biases)

        layer.wt_momentums = self.beta_1 * layer.wt_momentums + (1 - self.beta_1) * layer.der_weights
        layer.bs_momentums = self.beta_1 * layer.bs_momentums + (1 - self.beta_1) * layer.der_biases

        wt_momentums_corrected = layer.wt_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bs_momentums_corrected = layer.bs_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.wt_cache = self.beta_2 * layer.wt_cache + (1 - self.beta_2) * layer.der_weights**2
        layer.bs_cache = self.beta_2 * layer.bs_cache + (1 - self.beta_2) * layer.der_biases**2
        # Get corrected cache
        wt_cache_corrected = layer.wt_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bs_cache_corrected = layer.bs_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # SGD parameter update + normalization with square rooted cache
        layer.weights -= self.current_lr * wt_momentums_corrected / (np.sqrt(wt_cache_corrected) + self.eps)
        layer.biases -= self.current_lr * bs_momentums_corrected / (np.sqrt(bs_cache_corrected) + self.eps)

        # So, as beta approaches 0 (by raising it to the power of # iters), weight_momentums will approach der_weights here, 
        # thus the corrected momentum will approach der_weights concurrently.  The corrected caches will approach the squared 
        # der_weights. Thus the weights are incremented by (lr * der_weights) / (der_weights + epsilon) which is essentially 
        # just incrementing by the lr as we minimize loss (which itself approaches 0 due to decay).

    def post_update(self):
        self.iterations += 1
