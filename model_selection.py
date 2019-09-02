"""
module that has diferent functions to test and optimize models

expand_grid -- function to return the combinations of parameters to test a model
cv          -- function to cross validate a model
copy_model  -- return an object of the same class as the input model
tune_model  -- function to tune hyperparameters of a model given a grid of parameters 
"""

from itertools import product
from time import time

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

import numpy as np

import lightgbm as lgbm
import xgboost as xgb
import catboost as cat

from .handle_json import read_json, save_json


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


def cv(model, X, y, metric, folds = 5, split_type = 'normal', seed = None, verbose = 0):
    """
    Function to compute the cross validation metrics and get the out-of-sample predictions

    Parameters:
    model       -- a model object with a fit and predict methods
    X           -- pandas data frame containing the features used to train the model
    y           -- numpy array or pandas series containing the target variable
    metric      -- metric funtion used to evaluate the model. The function is of the form f(real, pred)
    folds       -- number of folds to use in the cross validation process
    split_type  -- can be 'normal', 'stratified' or 'timeseries'

    Return:
    metrics     -- a list with the evaluation results for each fold
    preds       -- a numpy array containg the out-of-sample predictions for each fold
    """
    # create a KFold object to compute fold indexes
    if split_type == 'stratified':
        kf = StratifiedKFold(n_splits = folds, random_state = seed)
        split = kf.split(X, y)
    elif split_type == 'timeseries':
        kf = TimeSeriesSplit(n_splits = folds)
        split = kf.split(X)
    else:
        kf = KFold(n_splits = folds, random_state = seed)
        split = kf.split(X)

    # create empty list to store the metrics
    metrics = []

    # create empty list to store the predictions
    preds = np.zeros(len(y))
    
    # for each fold, compute the metric and the out-of-sample prediction
    if verbose > 0:
        i = 0
        print("Initializing Cross-validation...\n")

    for id_train, id_val in split:
        if verbose > 0:
            tic = time()

        # fit the model with the train folds
        if type(model) == cat.core.CatBoostRegressor:
            # guardar features categoricos y su posición
            cat_features = [x for x in X.columns.values if X[x].dtypes == 'object']
            pos = [X.columns.get_loc(col) for col in cat_features]

            # crear el objeto pool y entrenar el modelo
            pool = cat.Pool(X.loc[id_train], y[id_train], cat_features = pos)
            model.fit(pool, verbose = 0)
        else:
            model.fit(X.loc[id_train], y[id_train])

        # predict with the out-of-sample fold

        # TODO: gran parche para que funcione con el LightGBM, arreglar cuanto antes
        preds[id_val] =  [i[1] for i in model.predict_proba(X.loc[id_val])]
        # compute the metric and append it to the metrics list
        metrics.append(metric(y[id_val], preds[id_val]))

        # print result
        if verbose > 0:
            i += 1
            toc = time()
            print(f"Finished fold {i} of {folds} folds. Result: {metric(y[id_val], preds[id_val])}. " +
            f"Time taken: {np.round((toc - tic)/60, 1)} m.")
            print("\n")

    return metrics, preds


def copy_model(model):
    """
    Check the type of model that is given as an imput, and return a new object of the 
    same class as the original model
    """

    # Light GBM
    if isinstance(model, lgbm.LGBMClassifier):
        return lgbm.LGBMClassifier(**model.get_params())

    # XGBoost
    elif isinstance(model, xgb.XGBClassifier):
        return xgb.XGBClassifier(**model.get_params())
    elif isinstance(model, xgb.XGBRegressor):
        return xgb.XGBRegressor(**model.get_params())

    else:
        print("Copy model not implemented yet")

def tune_model(model, X, y, params, metric, folds = 5, split_type = 'normal',  random_state = None, file_name = None):
    
    # get a tunning model
    tunning_model = copy_model(model)
    
    # get the grid of parameters to tune the model
    tune_params = expand_grid(params)
    
    # each tested params is a dictionary with the following structure {'params': {}, 'score': []}
    if file_name is not None:
        try:
            tested_params = read_json(file_name)
        except:
            tested_params = {'tested_params': []}
    else:
        tested_params = {'tested_params': []}
    
    # tune the model with differenct params
    for param in tune_params:
        tunning_model.set_params(**param)
        
        # check if the params are already tested
        tested_result = [result for result in tested_params['tested_params'] if result['params'] == tunning_model.get_params()]
        if len(tested_result) == 1:
            metrics = tested_result[0]['score']
        else:
            metrics, _ = cv(tunning_model, X, y, metric, folds = 5, split_type = split_type, seed = random_state)
            tested_params['tested_params'].append({'params': tunning_model.get_params(),
                                                   'score': metrics})
            
            # dump tested params into json file
            if file_name is not None:
                save_json(tested_params, file_name)

        # TODO: la función debería retornar los resultados para cada parámetro, no hacer print de nada
        print(f"Parameters: {param} | Score: {np.round(np.mean(metrics), 4)} +- {np.round(np.std(metrics), 5)}")
    
