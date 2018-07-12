from xgboost import XGBClassifier as XGBClass

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

    def fit(self, X, y):
        super(XGBClassifier, self).fit(X, y)

    def predict(self, X):
        return super(XGBClassifier, self).predict(X)
