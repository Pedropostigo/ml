import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, LGBMRegressor

def variable_importance(model, data):

    score = model.feature_importances_
    features = data.columns.values

    feat_importance = pd.DataFrame({'variables':  features,
                                    'importance': score})

    feat_importance = feat_importance.sort_values(by = 'importance', ascending = False)

    return feat_importance