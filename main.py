import numpy as np
import nnlayers
import nnloss
import nnoptimize
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

NUM_CLASSES = 3
NUM_EPOCHS = 10001
NUM_SAMPLES = 100
NUM_OUTPUTS = 2


def main():
    np.random.seed(0)
    X, y = spiral_data(NUM_SAMPLES, NUM_CLASSES)

    layers = (
        nnlayers.LayerDense(2, 64, wt_reg_l2=5e-4, bs_reg_l2=5e-4),
        nnlayers.LayerDense(64, NUM_CLASSES, activation="softmax")
    )
    loss = nnloss.Loss()
    optimizer = nnoptimize.AdamAdaptiveMomentum(learning_rate=0.05, decay=5e-5)

    # Test —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    losses = []
    for epoch in range(NUM_EPOCHS):
        # print(f"epoch {epoch}/{NUM_EPOCHS}", end="\r")
        # for batch in range(int(NUM_SAMPLES/BATCH_SIZE) + 1):
        #     sample_num =  batch * BATCH_SIZE
        #     sample_batch = X[sample_num : sample_num + BATCH_SIZE]
        #     true_batch = y[sample_num : sample_num + BATCH_SIZE]
            
        layers[0].forward(X)
        for layer_num in range(1, len(layers)):
            layers[layer_num].forward(layers[layer_num-1].outputs)
        if not epoch % 100:
            print("\nepoch:", epoch)
            describe(layers[-1], y, loss.batch_loss)
        
        loss.calc_loss(layers[-1].outputs, y, layers)
        losses.append(loss.batch_loss)

        layers[-1].backwards(loss.gradients)
        for layer_num in reversed(range(len(layers)-1)):
            layers[layer_num].backwards(layers[layer_num+1].gradients)

        optimizer.pre_update()
        for layer in layers:
            optimizer.update_params(layer)
        optimizer.post_update()

    # Validate ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    Xt, yt = spiral_data(NUM_SAMPLES, NUM_CLASSES)

    layers[0].forward(Xt)
    for layer_num in range(1, len(layers)):
        layers[layer_num].forward(layers[layer_num-1].outputs)

    loss.calc_loss(layers[-1].outputs, yt, layers)

    print("\nVALIDATION")
    describe(layers[-1], yt, loss.batch_loss)

    # LOSS PLOT ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
    plt.title("LOSS")
    plt.plot(losses)
    plt.show()


def describe(final_layer, y, loss):
    predictions = np.argmax(final_layer.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    print(f"acc: {accuracy} | Loss: {loss}")


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