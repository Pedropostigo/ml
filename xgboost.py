from xgboost import XGBClassifier as XGBClass
from xgboost import XGBRegressor as XGBReg
import pandas as pd
import numpy as np

# auxilar function to get the variable importance for both the XGB XGBClassifier
# and XGB Regressor
def variableImportance(model):
    # TODO: permitir cambiar el tipo de importancia
    # check if the model has been previously trained
    if model.isTrained:
        # get variable importance from original XGB Classifier
        varImp = model.get_booster().get_score(importance_type = 'weight')
        # save variable importance in a Data Frame
        varImp = pd.DataFrame({'variables': list(varImp.keys()),
                                'importance': list(varImp.values())})
        # reorder variables in the data frame
        varImp = varImp.reindex(('variables', 'importance'), axis = 1)
        # compute importance as percentage over total
        varImp['importance'] = varImp['importance']/np.sum(varImp['importance'])
        # sort variables from most important to least important
        varImp = varImp.sort_values(by = 'importance', ascending = False)
        return varImp

    else:
        # TODO: raise an exception saying that the model is not trained
        print("Model is not trained. Model must be trained first")


# class for the XGB Classifier where the methods will be added
class XGBClassifier(XGBClass):
    def __init__(self,
                max_depth = 3,
                learning_rate = 0.1,
                n_estimators = 100,
                silent = True,
                objective = 'binary:logistic',
                booster = 'gbtree',
                n_jobs = 1,
                nthread = None,
                gamma = 0,
                min_child_weight = 1,
                max_delta_step = 0,
                subsample = 1,
                colsample_bytree = 1,
                colsample_bylevel = 1,
                reg_alpha = 0,
                reg_lambda = 0,
                scale_pos_weight = 1,
                base_score = 0.5,
                random_state = 0,
                seed = None,
                missing = None,
                **kwargs):

        # call the original __init__ method of the XGBClassifier
        super(XGBClassifier, self).__init__(max_depth, learning_rate, n_estimators,
                                            silent, objective, booster, n_jobs,
                                            nthread, gamma, min_child_weight,
                                            max_delta_step, subsample,
                                            colsample_bytree, colsample_bylevel,
                                            reg_alpha, reg_lambda, scale_pos_weight,
                                            base_score, random_state, seed,
                                            missing, **kwargs)

        # variable to save if the model has been previously trained
        self.isTrained = False

    def fit(self, X, y):
        super(XGBClassifier, self).fit(X, y)
        #save that the model has been trained
        self.isTrained = True

    def predict(self, X):
        return super(XGBClassifier, self).predict(X)

    def variableImportance(self):
        return(variableImportance(self))


# class for the CGB Regressor where the methods will be added
class XGBRegressor(XGBReg):
    def __init__(self,
                max_depth = 3,
                learning_rate = 0.1,
                n_estimators = 100,
                silent = True,
                objective ='reg:linear',
                booster = 'gbtree',
                n_jobs = 1,
                nthread = None,
                gamma = 0,
                min_child_weight = 1,
                max_delta_step = 0,
                subsample = 1,
                colsample_bytree = 1,
                colsample_bylevel = 1,
                reg_alpha = 0,
                reg_lambda = 1,
                scale_pos_weight = 1,
                base_score = 0.5,
                random_state = 0,
                seed = None,
                missing = None,
                **kwargs):
        # call the original __init__ method of the XGBClassifier
        super(XGBRegressor, self).__init__(max_depth, learning_rate, n_estimators,
                                            silent, objective, booster, n_jobs,
                                            nthread, gamma, min_child_weight,
                                            max_delta_step, subsample, colsample_bytree,
                                            colsample_bylevel, reg_alpha, reg_lambda,
                                            scale_pos_weight, base_score, random_state,
                                            seed, missing, **kwargs)

        self.isTrained = False

    def fit(self, X, y):
        super(XGBRegressor, self).fit(X, y)
        # save that the model has been trained
        self.isTrained = True

    def predict(self, X):
        return super(XGBRegressor, self).predict(X)

    def variableImportance(self):
        return(variableImportance(self))
