import numpy as np

from sklearn.metrics import make_scorer

# Root mean squared error
def rmse(real, pred):
    return(np.sqrt((1/len(real)) * np.sum(np.square(real-pred))))

# Symmetric mean absolute percentage error
def smape(real, pred):
    return (100 / len(real) * np.sum(np.abs(pred-real)/((np.abs(real)+np.abs(pred))/2)))

def score(metric):
    """
    Fucntion to get a scorer
    """
    if metric == "rmse":
        return(make_scorer(rmse, greater_is_better = False))
    elif metric == 'smape':
        return(make_scorer(smape, greater_is_better = False))
