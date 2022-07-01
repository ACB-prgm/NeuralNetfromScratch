from typing import final
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnoptimize
import nnlayers
import nnloss
import pathlib
import pickle
import os

BASE_DIR = pathlib.Path(__file__).resolve().parent
PICKLES_PATH = os.path.join(BASE_DIR, "NeuralNetfromScratch/pickles")
NUM_EPOCHS = 10001
NUM_SAMPLES = 100
NUM_CLASSES = 2
NUM_OUTPUTS = 2


def main():
    np.random.seed(0)
    X, y = spiral_data(NUM_SAMPLES, NUM_CLASSES)
    y = y.reshape(-1, 1)

    layers = (
        nnlayers.LayerDense(2, 64, wt_reg_l2=5e-4, bs_reg_l2=5e-4),
        nnlayers.LayerDense(64, 1, activation="sigmoid")
    )
    loss = nnloss.Loss(func="binary_cross_entropy")
    optimizer = nnoptimize.AdamAdaptiveMomentum(learning_rate=0.001, decay=5e-7)

    # TRAIN —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    losses = []
    for epoch in range(NUM_EPOCHS):
        # print(f"epoch {epoch}/{NUM_EPOCHS}", end="\r")
        # for batch in range(int(NUM_SAMPLES/BATCH_SIZE) + 1):
        #     sample_num =  batch * BATCH_SIZE
        #     sample_batch = X[sample_num : sample_num + BATCH_SIZE]
        #     true_batch = y[sample_num : sample_num + BATCH_SIZE]

        # FORWARD PASS    
        layers[0].forward(X)
        for layer_num in range(1, len(layers)):
            layers[layer_num].forward(layers[layer_num-1].outputs)
        
        # LOSS CALCULATION
        loss.calc_loss(layers[-1].outputs, y, layers)
        losses.append(loss.batch_loss)

        if not epoch % 100:
            print("\nepoch:", epoch)
            describe(layers[-1], y, loss.batch_loss)

        # BACKWARD PASS
        layers[-1].backwards(loss.gradients)
        for layer_num in reversed(range(len(layers)-1)):
            layers[layer_num].backwards(layers[layer_num+1].gradients)

        optimizer.pre_update()
        for layer in layers:
            optimizer.update_params(layer)
        optimizer.post_update()

    # Validate ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    Xt, yt = spiral_data(NUM_SAMPLES, NUM_CLASSES)
    yt = y.reshape(-1, 1)

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
    if final_layer.activation == "sigmoid":
        predictions = (final_layer.outputs > 0.5) * 1
    else:    
        predictions = np.argmax(final_layer.outputs, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    print(f"acc: {accuracy} | Loss: {loss}")


if __name__ == "__main__":
    main()