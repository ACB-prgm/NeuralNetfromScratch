import numpy as np
import pickle
import os


BASE_DIR = os.path.dirname(__file__)



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

spanish_data()