"""
functions to manipulate data frames
"""

import numpy as np
import pandas as pd

def feature_inspection(data, unique_vals = 5):
    """
    Function to inspect features of a data frame.
    Returns a data frame with feature names, type of feature and unique values

    Parameters:
    unique_vals -- number of unique values per feature to show in the results

    Return:

    """
    features = data.columns.values
    types = []
    num_unique_values = []
    unique_values = []
    num_missing = []
    
    for feat in features:
        types.append(data[feat].dtype.name)
        
        # if feature is categorical, get categories, else get unique values
        if data[feat].dtype.name == 'category':
            unique_values.append(list(data[feat].cat.categories)[0:unique_vals])
        else:
            unique_values.append(data[feat].unique()[1:unique_vals])
            
        # if feature is categorical or object, get number of unique values
        if data[feat].dtype.name == 'category':
            num_unique_values.append(len(list(data[feat].cat.categories)))
        elif data[feat].dtype.name == 'object':
            num_unique_values.append(len(data[feat].unique()))
        else:
            num_unique_values.append(np.nan)

        # number of missing values in feature
        num_missing.append(np.sum(data[feat].isna() * 1))
        
    result = pd.DataFrame({'feature': features,
                            'type': types,
                            'num_unique_values': num_unique_values,
                            'unique_values': unique_values,
                            'num_missing': num_missing})
    return result