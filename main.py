from cProfile import label
import matplotlib.pyplot as plt
import nnfs.datasets as nnfsds
import numpy as np
import nnoptimize
import nnlayers
import pathlib
import nnloss
import pickle
import nnfs
import os

BASE_DIR = pathlib.Path(__file__).resolve().parent
PICKLES_PATH = os.path.join(BASE_DIR, "NeuralNetfromScratch/pickles")
NUM_EPOCHS = 1000
NUM_SAMPLES = 100
NUM_CLASSES = 2
NUM_OUTPUTS = 2

def main():
    nnfs.init()
    np.random.seed(0)
    X, y = nnfsds.spiral_data(NUM_SAMPLES, NUM_CLASSES)
    # plt.scatter(X[:,0], y)
    # plt.show()
    # quit()
    # X,y = nnfsds.sine_data()
    # y = y.reshape(-1, 1) # for binary regression

    layers = (
        nnlayers.LayerDense(2, 64),
        nnlayers.LayerDense(64, 2, activation="softmax")
    )
    loss = nnloss.Loss(func="categorical_cross_entropy")
    optimizer = nnoptimize.AdamAdaptiveMomentum(learning_rate=0.01, decay=1e-6) # decay=5e-7

    # TRAIN —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    losses = []
    for epoch in range(NUM_EPOCHS+1):
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
            print("epoch:", epoch)
            describe(layers[-1], y, loss.batch_loss, optimizer)

        # BACKWARD PASS
        layers[-1].backwards(loss.gradients)
        for layer_num in reversed(range(len(layers)-1)):
            layers[layer_num].backwards(layers[layer_num+1].gradients)

        optimizer.pre_update()
        for layer in layers:
            optimizer.update_params(layer)
        optimizer.post_update()

    plt.scatter(X[:, 0], y, label="TRUE")
    plt.scatter(X[:, 0], layers[-1].outputs[:, 0], label="PRED")
    plt.title("TRAINING DATASET")
    plt.legend()
    plt.show()
    plt.cla()

    # Validate ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    Xt, yt = nnfsds.spiral_data(NUM_SAMPLES, NUM_CLASSES)
    # yt = y.reshape(-1, 1)
    # Xt, yt = nnfsds.sine_data()

    layers[0].forward(Xt)
    for layer_num in range(1, len(layers)):
        layers[layer_num].forward(layers[layer_num-1].outputs)

    loss.calc_loss(layers[-1].outputs, yt, layers)

    print("\nVALIDATION")
    describe(layers[-1], yt, loss.batch_loss, optimizer)

    plt.scatter(Xt[:, 0], yt, label="TRUE")
    plt.scatter(Xt[:, 0], layers[-1].outputs[:, 0], label="PRED")
    plt.title("VALIDATION DATASET")
    plt.legend()
    plt.show()

    # LOSS PLOT ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # plt.title("LOSS")
    # plt.plot(losses)
    # plt.show()


def describe(final_layer, y, loss, optimizer):
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

    print(f"acc: {accuracy:.3f} | Loss: {loss:.3f} | lr: {optimizer.current_lr:.3f}")


if __name__ == "__main__":
    main()