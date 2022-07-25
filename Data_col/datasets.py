import pandas as pd
import numpy as np
import pickle
import os


BASE_DIR = os.path.dirname(__file__)
MNIST_DIR = os.path.join(BASE_DIR, "FashionMNIST")


def spanish_data():
    FILE = "spanish_nouns.pickle"
    with open(os.path.join(BASE_DIR, FILE), "rb") as file:
        raw = pickle.load(file)
    
    THES = ["el", "la"]
    X = []
    y = []
    for word in raw:
        word_split = word.split(" ")
        if (any(map(word.__contains__, ("el/la", "del ", "al ", "los ", "las ")))) or len(word_split) > 2 \
            or len(word_split[-1]) < 3:
            continue
        
        X.append([ord(char) / ord("z") for char in list(word_split[-1][-3:])]) # float value for each character
        y.append(THES.index(word_split[0]))
    
    to_remove = int(y.count(0) < y.count(1))
    for _ in range(abs(y.count(0) - y.count(1))): # balance the classes / make same length
        idx = y.index(to_remove)
        del X[idx]
        del y[idx]
    
    y, X = (zip(* sorted(zip(y, X))))
    
    X = list(X)
    y = list(y)

    nval = int(len(X) * .1) # 10% of size of the array
    Xt, yt = X[:nval] + X[-nval:], y[:nval] + y[-nval:] # test data == 10% of the data, 5% from each end

    del X[:nval]
    del X[-nval:]
    del y[:nval]
    del y[-nval:]

    data = [X, y, Xt, yt]

    for idx, datum in enumerate(data):
        data[idx] = np.array(datum)

    return data

def fashion_MNIST():
    BASE_FILENAME = "fashion-mnist_X.csv"

    df = pd.read_csv(os.path.join(MNIST_DIR, BASE_FILENAME.replace("X", "train")))
    df2 = pd.read_csv(os.path.join(MNIST_DIR, BASE_FILENAME.replace("X", "train_2")))
    df = pd.concat((df, df2)).reset_index().drop(df.columns[0], axis=1).apply(pd.to_numeric)
    X, y = MNIST_from_df(df)

    df = pd.read_csv(os.path.join(MNIST_DIR, BASE_FILENAME.replace("X", "test"))).apply(pd.to_numeric)
    Xt, yt = MNIST_from_df(df)

    return X, y, Xt, yt


def MNIST_from_df(df):
    y = df.pop("label")
    y = np.eye(len(y), M=10)[y]
    X = (df.to_numpy(dtype=np.float32) - 127.5) / 127.5 # max is 255, so this will limit values -1 < X < 1

    return X, y