from catboost import CatBoostRegressor as _CatBoostRegressor
from catboost import CatBoostClassifier as _CatBoostClassifier

"""
Module with a CatBoost object, and object to wrap all the utility functions
to model using CatBoost
"""

def catboost_feat_importance(model, data):

    # compute the prediction value change importance and the 
    # loss change importance
    pred_change_importance = model.get_feature_importance(data = data,
                                                        type = 'PredictionValuesChange',
                                                        prettified = True)

    loss_change_importance = model.get_feature_importance(data = data,
                                                        type = 'LossFunctionChange',
                                                        prettified = True)

    # merge the two imporances computations in a data frame
    importances = pred_change_importance.merge(loss_change_importance, how = 'left', 
                                                on = 'Feature Id')
    importances = importances.rename(columns = {'Importances_x': 'PredValChange Importance', 
                                                'Importances_y': 'LossChange Importance'})

    return importances


class CatBoostRegressor(_CatBoostRegressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class CatBoostClassifier(_CatBoostClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    import os
    import pathlib
    MODULE_ABSPATH = pathlib.Path(__file__).parent.parent.resolve()

    import pandas as pd
    
    data = pd.read_csv(os.path.join(MODULE_ABSPATH, "datasets", "california_housing", 
                        "california_housing.zip"))

    cat_regressor = CatBoostRegressor(random_seed = 1234, num_trees = 100)
    print('CATBOOST REGRESSOR ...')
    print(cat_regressor.get_params())

    print()

    cat_regressor = CatBoostClassifier(random_seed = 4321, num_trees = 200)
    print('CATBOOST CLASSIFIER ...')
    print(cat_regressor.get_params())