from types import new_class
from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LinearRegression
import openml
import pandas as pd

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LinRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LinearRegression) -> LinRegParams:
    """Returns the paramters of a sklearn LinearRegression model."""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,]
    return params


def set_model_params(
    model: LinearRegression, params: LinRegParams
) -> LinearRegression:
    """Sets the parameters of a sklean LinearRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LinearRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LinearRegression documentation for more
    information.
    """
    #n_classes = 1
    n_features = 5 # Number of features in dataset
    #model.classes_ = np.array([i for i in range(1)])

    model.coef_ = np.zeros((n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))


def load_data() -> Dataset:
    data = pd.read_csv('data.csv')
    data.reset_index(drop=True)
    df =np.array(data)
    X = df[:,:]
    y =df[:,1]
    y=y.astype('int')
    """ Select the 70% of the data as Training data and 30% as test data """
    x_train, y_train = X[:10500], y[:10500]
    x_test, y_test = X[10500:], y[10500:]
    return (x_train, y_train), (x_test, y_test)
        


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )