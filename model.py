import numpy as np


class Model:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.history = []
    
    def add_layer(self, layer_dense_obj):
        self.layers.append(layer_dense_obj)
    
    def set_loss(self, loss_obj):
        self.loss = loss_obj

    def set_optimizer(self, optimizer_obj):
        self.optimizer = optimizer_obj

    def forward(self, X, y, training=False):
        self.layers[0].forward(X, training)
        for layer_num in range(1, len(self.layers)):
            self.layers[layer_num].forward(self.layers[layer_num-1].outputs, training)
        
        self.loss.calc_loss(self.layers[-1].outputs, y, self.layers)

    def backward(self, training=False):
        self.layers[-1].backwards(self.loss.gradients, training)
        for layer_num in reversed(range(len(self.layers)-1)):
            self.layers[layer_num].backwards(self.layers[layer_num+1].gradients, training)

        if training:
            self.optimizer.pre_update()
            for layer in self.layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update()

    def train(self, X, y, num_epochs=1, print_every=100):
        for epoch in range(num_epochs+1):
            self.forward(X, y, training=True)
            self.backward(training=True)

            if print_every and not epoch % print_every:
                print("epoch:", epoch)
                self.stats(y)
    
    def validate(self, Xt, yt):
        self.forward(Xt, yt, training=False)
        print("\nVALIDATION")
        self.stats(yt)

    def stats(self, y):
        final_layer = self.layers[-1]
        if final_layer.activation == "softmax":    
            predictions = np.argmax(final_layer.outputs, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions==y)
    
        elif final_layer.activation == "sigmoid":
            predictions = (final_layer.outputs > 0.5) * 1
            accuracy = np.mean(predictions==y)

        elif final_layer.activation == "linear": # used with scalar values so must calc acc differently
            tolerance = np.std(y) # tolerance or allowance is how much error is allowed (250 is arbitrary)
            predictions = final_layer.outputs
            accuracy = np.mean(np.absolute(predictions - y) < tolerance) # acc = the mean of the differences that are within the tolerance

        print(f"acc: {accuracy:.3f} \
        | Loss: {self.loss.batch_loss:.3f} \
        | lr: {self.optimizer.current_lr:.3f}")