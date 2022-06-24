import numpy as np
import nnlayers
import nnbprop


CLASS_SIZE = 3
BATCH_SIZE = 3
NUM_OUTPUTS = 2


def main():
    np.random.seed(0)
    X, y = create_data(100, CLASS_SIZE, D=2)

    print(X[:10], y[:10])
    layer_1 = nnlayers.LayerDense(2, 10)
    layer_2 = nnlayers.LayerDense(layer_1.num_neurons, 10)
    layer_3 = nnlayers.LayerDense(layer_2.num_neurons, 3)

    layer_1.forward(X[:BATCH_SIZE]) # doing abatch of 3 for readability
    layer_1.apply_ReLU()

    layer_2.forward(layer_1.outputs)
    layer_2.apply_ReLU()

    layer_3.forward(layer_2.outputs)
    layer_3.apply_softmax()

    loss = nnbprop.Loss(layer_3.outputs, y[:BATCH_SIZE])
    loss.calc_loss()

    print(loss.batch_loss)


def create_data(N, K, D=2): # taken from https://cs231n.github.io/neural-networks-case-study/
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    
    return X, y



if __name__ == "__main__":
    main()