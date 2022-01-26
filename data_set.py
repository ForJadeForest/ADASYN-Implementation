from adasyn import adasyn, my_adasyn
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def process_data_set(data_set_name, path):
    if data_set_name == 'vehicle':
        return _process_vehicle(path)
    elif data_set_name == 'archive':
        return _process_archive(path)
    elif data_set_name == 'ionosphere':
        return _process_ionosphere(path)
    elif data_set_name == 'vol':
        return _process_vol(path)
    elif data_set_name == 'abalone':
        return _process_abalone(path)


def _process_vehicle(path):
    file_data_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_data_list.append(pd.read_csv(file_path, names=list(range(0, 20)), sep=' '))
    data = pd.concat([i for i in file_data_list])
    data = data.reset_index(drop=True)
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]

    def remake(x):
        if x != 'van':
            x = 'other'
        return x

    y = y.apply(remake)
    labels = y.value_counts()
    X = X.to_numpy()
    y = y.to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    new_x, new_y = adasyn(Xtrain, Ytrain, labels)
    return (new_x, new_y), (Xtrain, Ytrain), (Xtest, Ytest)


def _process_archive(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    labels = y.value_counts()
    X = X.to_numpy()
    y = y.to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    new_x, new_y = adasyn(Xtrain, Ytrain, labels)
    return ((new_x, new_y), (Xtrain, Ytrain), (Xtest, Ytest))


def _process_ionosphere(path):
    data = pd.read_csv(path, names=list(range(0, 35)))
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    labels = y.value_counts()
    X = X.to_numpy()
    y = y.to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    new_x, new_y = adasyn(Xtrain, Ytrain, labels)
    return (new_x, new_y), (Xtrain, Ytrain), (Xtest, Ytest)


def _process_vol(path):
    data = pd.read_csv(path, names=list(range(0, 14)), sep=' ')
    X = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    def remake(x):
        if x != 1:
            x = 0
        return x
    y = y.apply(remake)
    labels = y.value_counts()
    X = X.to_numpy()
    y = y.to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    new_x, new_y = adasyn(Xtrain, Ytrain, labels)
    return (new_x, new_y), (Xtrain, Ytrain), (Xtest, Ytest)


def _process_abalone(path):
    data = pd.read_csv(path, names=list(range(0, 10)))
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    labels = y.value_counts()
    X = X.to_numpy()
    y = y.to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    new_x, new_y = adasyn(Xtrain, Ytrain, labels)
    return (new_x, new_y), (Xtrain, Ytrain), (Xtest, Ytest)