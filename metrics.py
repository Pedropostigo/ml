import numpy as np

from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score

# Root mean squared error
def rmse(real, pred):
    return np.sqrt(mean_squared_error(real, pred))

# Symmetric mean absolute percentage error
def smape(real, pred):
    return (100 / len(real) * np.sum(np.abs(pred-real)/((np.abs(real)+np.abs(pred))/2)))

# Area under the ROC curve
def auc(real, pred):
    return roc_auc_score(real, pred)

def score(metric):
    """
    Function to get a scorer
    """
    if metric == "rmse":
        return(make_scorer(rmse, greater_is_better = False))
    elif metric == 'smape':
        return(make_scorer(smape, greater_is_better = False))
