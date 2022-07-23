import numpy as np
import pickle
import os


BASE_DIR = os.path.dirname(__file__)



def get_spanish_data():
    FILE = "spanish_nouns.pickle"
    with open(os.path.join(BASE_DIR, FILE), "rb") as file:
        raw = pickle.load(file)
    
    THES = ["el", "la"]
    X = []
    y = []
    for word in raw:
        word_split = word.split(" ")
        if (any(map(word.__contains__, ("el/la", "del ", "al ", "los ", "las ")))) or len(word_split) > 2:
            continue
        
        X.append([ord(char) / ord("z") for char in list(word_split[-1][-3:])])
        y.append(THES.index(word_split[0]))
    
    y, X = (zip(* sorted(zip(y, X))))
    X = list(X)
    y = list(y)

    for idx, ls in enumerate(X):
        if len(ls) < 3:
            del X[idx]
            del y[idx]

    nval = int(len(X) * .1)
    Xt, yt = X[:nval] + X[-nval:], y[:nval] + y[-nval:]

    del X[:nval]
    del X[-nval:]
    del y[:nval]
    del y[-nval:]

    data = [X, y, Xt, yt]

    for idx, datum in enumerate(data):
        data[idx] = np.array(datum)

    return data