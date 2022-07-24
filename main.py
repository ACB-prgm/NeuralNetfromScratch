import Data_col.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import nnoptimize
import nnlayers
import pathlib
import nnloss
import pickle
import model
import nnfs
import os

BASE_DIR = pathlib.Path(__file__).resolve().parent
PICKLES_PATH = os.path.join(BASE_DIR, "NeuralNetfromScratch/pickles")
NUM_EPOCHS = 5000
NUM_SAMPLES = 100
NUM_CLASSES = 3
NUM_OUTPUTS = 2

def main():
    X, y , Xt, yt = datasets.spanish_data()


    NN = model.Model()

    NN.add_layer(nnlayers.LayerDense(NUM_CLASSES, 64, wt_reg_l2=5e-4, bs_reg_l2=5e-4))
    NN.add_layer(nnlayers.LayerDense(64, 64, wt_reg_l2=5e-4, bs_reg_l2=5e-4))
    NN.add_layer(nnlayers.LayerDense(64, 2, activation="softmax"))
    NN.set_loss(nnloss.Loss(func="categorical_cross_entropy"))
    NN.set_optimizer(nnoptimize.AdamAdaptiveMomentum(learning_rate=0.01, decay=1e-6))

    nnfs.init()
    np.random.seed(0)
    # X, y = nnfsds.spiral_data(NUM_SAMPLES, NUM_CLASSES)

    NN.train(X, y, NUM_EPOCHS)

    plt.scatter(X[:, 0], y, label="TRUE")
    plt.scatter(X[:, 0], NN.layers[-1].outputs[:, 0], label="PRED")
    plt.title("TRAINING DATASET")
    plt.legend()
    plt.show()
    plt.cla()

    # Xt, yt = nnfsds.spiral_data(NUM_SAMPLES, NUM_CLASSES)
    NN.validate(Xt, yt)

    plt.scatter(Xt[:, 0], yt, label="TRUE")
    plt.scatter(Xt[:, 0], NN.layers[-1].outputs[:, 0], label="PRED")
    plt.title("VALIDATION DATASET")
    plt.legend()
    plt.show()

    # LOSS PLOT ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # plt.title("LOSS")
    # plt.plot(losses)
    # plt.show()


if __name__ == "__main__":
    main()
