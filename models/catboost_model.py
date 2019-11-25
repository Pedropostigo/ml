from catboost import Pool, CatBoostRegressor

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
    importances = pred_change_importance.merge(loss_change_importance, how = 'left', on = 'Feature Id')
    importances = importances.rename(columns = {'Importances_x': 'PredValChange Importance', 'Importances_y': 'LossChange Importance'})

    return importances