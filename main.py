import numpy as np
import nnlayers
import nnbprop


NUM_INPUTS = 2
NUM_OUTPUTS = 2


def main():
    np.random.seed(0)
    X, y = create_data(100, NUM_INPUTS)
    
    layer_1 = nnlayers.LayerDense(NUM_INPUTS, 10)
    layer_2 = nnlayers.LayerDense(layer_1.num_neurons, 10)
    layer_3 = nnlayers.LayerDense(layer_2.num_neurons, 3)

    layer_1.forward(X[:3]) # doing abatch of 3 for readability
    layer_1.apply_ReLU()

    layer_2.forward(layer_1.outputs)
    layer_2.apply_ReLU()

    layer_3.forward(layer_2.outputs)
    layer_3.apply_softmax()

    loss = nnbprop.Loss(layer_3.outputs, y)
    loss.calc_loss()

    print(loss.batch_loss)


def create_data(points, classes): # taken from https://cs231n.github.io/neural-networks-case-study/
    X = np.zeros((points * classes, 2)) # data matrix (each row = single example)
    y = np.zeros(points * classes, dtype = 'uint8') # class labels
    for j in range(classes):
        ix = range(points * j, points * (j + 1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(j * 4,(j + 1) * 4, points) + np.random.randn(points) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    
    return X, y



if __name__ == "__main__":
    main()