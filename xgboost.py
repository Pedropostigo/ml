from xgboost import XGBClassifier as XGBClass
from xgboost import XGBRegressor as XGBReg

import pandas as pd
import numpy as np

# auxilar function to get the variable importance for both the XGB XGBClassifier
# and XGB Regressor
def variableImportance(model, sort):

    # check if the model has been previously trained
    if model.isTrained:
        # get variable importance from original XGB Classifier
        weight = model.get_booster().get_score(importance_type = 'weight')
        gain = model.get_booster().get_score(importance_type = 'gain')
        cover = model.get_booster().get_score(importance_type = 'cover')

        # save variable importance in a Data Frame

        # weight = number of times a particular feature occurs in the trees of the model
        weight = pd.DataFrame({'variables': list(weight.keys()),
                                'weight': list(weight.values())})

        # gain = feature contribution for each tree in the model. more gain -> more important
        # in generating a prediction
        gain = pd.DataFrame({'variables': list(gain.keys()),
                                'gain': list(gain.values())})

        # coverage = relative number of observations related to the feature
        cover = pd.DataFrame({'variables': list(cover.keys()),
                                'cover': list(cover.values())})

        # union of data frames in one final dataframe
        varImp = pd.merge(weight, gain, how = "outer", on = "variables")
        varImp = pd.merge(varImp, cover, how = "outer", on = "variables")


        # reorder variables in the data frame
        varImp = varImp.reindex(('variables', 'gain', 'cover', 'weight'), axis = 1)

        # compute importance as percentage over total
        # varImp['importance'] = varImp['importance']/np.sum(varImp['importance'])
        
        # sort variables from most important to least important
        varImp = varImp.sort_values(by = sort, ascending = False)
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
                missing = None,
                **kwargs):

        # call the original __init__ method of the XGBClassifier
        super(XGBClassifier, self).__init__(max_depth = max_depth, learning_rate = learning_rate, 
                                            n_estimators = n_estimators, silent = silent, 
                                            objective = objective, booster = booster, n_jobs = n_jobs,
                                            nthread = nthread, gamma = gamma, min_child_weight = min_child_weight,
                                            max_delta_step = max_delta_step, subsample = subsample,
                                            colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel,
                                            reg_alpha = reg_alpha, reg_lambda = reg_lambda, scale_pos_weight = scale_pos_weight,
                                            base_score = base_score, random_state = random_state,
                                            missing = missing, **kwargs)

        # variable to save if the model has been previously trained
        self.isTrained = False

    def fit(self, X, y, eval_set = None, eval_metric = None, verbose = False):
        super(XGBClassifier, self).fit(X, y, eval_set = eval_set, eval_metric = eval_metric, verbose = verbose)
        #save that the model has been trained
        self.isTrained = True

    def evals_result(self):
        """
        Method that returns the evaluation sets given in fit method
        """
        return  super(XGBClassifier, self).evals_result()

    def predict(self, X, probs = True):
        # TODO: solve what to do when predicting multiple classes
        if probs:
            preds = super(XGBClassifier, self).predict_proba(X)
            preds = [pred[1] for pred in preds]
            return preds
        else:
            return super(XGBClassifier, self).predict(X)

    def variableImportance(self):
        return(variableImportance(self, sort = "gain"))


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
                missing = None,
                **kwargs):
        # call the original __init__ method of the XGBRegressor
        super(XGBRegressor, self).__init__(max_depth = max_depth, learning_rate = learning_rate, 
                                            n_estimators = n_estimators, silent = silent,
                                            objective = objective, booster = booster, n_jobs = n_jobs,
                                            nthread = nthread, gamma = gamma, min_child_weight = min_child_weight,
                                            max_delta_step = max_delta_step, subsample = subsample, 
                                            colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, 
                                            reg_alpha = reg_alpha, reg_lambda = reg_lambda,
                                            scale_pos_weight = scale_pos_weight, base_score = base_score, 
                                            random_state = random_state, missing = missing, **kwargs)

        self.isTrained = False

    def fit(self, X, y, eval_set = None, eval_metric = None, verbose = False):
        super(XGBRegressor, self).fit(X, y, eval_set = eval_set, eval_metric = eval_metric, verbose = verbose)
        # save that the model has been trained
        self.isTrained = True
    
    def evals_result(self):
        """
        Method that returns the evaluation sets given in fit method
        """
        return  super(XGBRegressor, self).evals_result()


    def predict(self, X):
        return super(XGBRegressor, self).predict(X)

    def variableImportance(self):
        return(variableImportance(self, sort = "gain"))

    def tune(self, X, y, params, metric, cv = 5):
        from sklearn.model_selection import GridSearchCV
        from metrics import score

        gs = GridSearchCV(self, params, cv = cv, scoring = score(metric))
        gs.fit(X, y)

        print(f"Best score: {gs.best_score_}")
        print(f"Best parameters: {gs.best_params_}")
