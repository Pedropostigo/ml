from itertools import product

from sklearn.model_selection import KFold

import numpy as np


def expand_grid(params):
    """
    Function to compute all the combinations of a lists of parameters

    Parameters:
    params  -- dict with the parameter names as keys and list of parameter values as values

    Return:
    grid    -- a list of dicts containing the combination of parameters
    """

    # list of parameters
    keys = list(params.keys())

    # get all combinations of parameters
    comb = []
    for param in params:
        comb.append(params[param])

    comb = list(product(*comb))

    # list to store all dicts containing all combinations of parameters
    grid = []

    # create each dict containing each combiantion of parameters
    for i in range(0, len(comb)):
        # create a dictionary to store the parameters
        par = {}
        
        # pupulate the dictionary with the parameters
        for j in range(0, len(keys)):
            par[keys[j]] = comb[i][j]

        # append the parameters to the grid
        grid.append(par)

    return grid


def cv(model, X, y, metric, folds = 5):
    """
    Function to compute the cross validation metrics and get the out-of-sample predictions

    Parameters:
    model   -- a model object with a fit and predict methods
    X       -- pandas data frame containing the features used to train the model
    y       -- numpy array or pandas series containing the target variable
    metric  -- metric funtion used to evaluate the model. The function is of the form f(real, pred)
    folds   -- number of folds to use in the cross validation process

    Return:
    metrics -- a list with the evaluation results for each fold
    preds   -- a numpy array containg the out-of-sample predictions for each fold
    """
    # create a KFold object to compute fold indexes
    kf = KFold(n_splits = folds)

    # create empty list to store the metrics
    metrics = []

    # create empty list to store the predictions
    preds = np.zeros(len(y))

    # for each fold, compute the metric and the out-of-sample prediction
    for id_train, id_val in kf.split(X):
        # fit the model with the train folds
        model.fit(X.loc[id_train], y[id_train])
        # predict with the out-of-sample fold
        preds[id_val] =  model.predict(X.loc[id_val])
        # compute the metric and append it to the metrics list
        metrics.append(metric(y[id_val], preds[id_val]))

    return metrics, preds