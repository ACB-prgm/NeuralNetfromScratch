import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os


BASE_DIR = os.path.dirname(__file__)
TEST_MNIST = os.path.join(BASE_DIR, "FashionMNIST/fashion-mnist_test.csv")



def main():
    show_fashion()
    # TRAIN = "FashionMNIST/fashion-mnist_train.csv"
    # df = pd.read_csv(os.path.join(BASE_DIR, TRAIN)).apply(pd.to_numeric)

    # length = len(df)
    # df2 = df.iloc[math.floor(length/2):]
    # df = df.iloc[:math.ceil(length/2)]

    # df2.to_csv(os.path.join(BASE_DIR, TRAIN.replace(".", "_2.")))
    # df.to_csv(os.path.join(BASE_DIR, TRAIN))


def show_fashion():
    df = pd.read_csv(TEST_MNIST).apply(pd.to_numeric)
    df = df.drop(df.columns[0], axis=1)

    ser = df.iloc[10]

    values = list(ser)
    coords = np.array([get_coords(idx) for idx, _value in enumerate(values)])

    plot = plt.scatter(coords[:, 0], coords[:, 1], c=values)
    plt.show()


def get_coords(pixel_num: int):
    x = pixel_num % 28
    y = abs((pixel_num // 28) - 28)

    return [x, y]


if __name__ == "__main__":
    main()

