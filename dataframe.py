"""
functions to manipulate data frames
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

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
            unique_values.append(data[feat].unique()[0:unique_vals])
            
        # if feature is categorical or object, get number of unique values
        if data[feat].dtype.name == 'category':
            num_unique_values.append(len(list(data[feat].cat.categories)))
        else:
            num_unique_values.append(len(data[feat].unique()))

        # number of missing values in feature
        num_missing.append(np.sum(data[feat].isna() * 1))
        
    result = pd.DataFrame({'feature': features,
                            'type': types,
                            'num_unique_values': num_unique_values,
                            'unique_values': unique_values,
                            'num_missing': num_missing})
    return result


def reduce_mem_usage(df, progress_bar = True, verbose = True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2

    # a progress bar is shown tracking how many features are left
    if progress_bar:
        columns = tqdm(df.columns)
    else:
        columns = df.columns
    
    for col in columns:
        col_type = df[col].dtype.name
        
        if col_type not in ['object', 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f"Memory usage: start = {np.round(start_mem, 2)}, end = {np.round(end_mem, 2)}, " + \
            f"decreased by = {np.round((100 * (start_mem - end_mem) / start_mem),1)} %")
    
    return df