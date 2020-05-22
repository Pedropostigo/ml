import numpy as np
import pandas as pd

def histogram(X, feat, n_buckets = 10, scale = False, left_outliers = None, 
                right_outliers = None, agg_params = None):
    """
    Function to compute the histogram of a variable, and different agregation meassures for the rest of the columns
    
    Parameters
    -----------------
    X (DataFrame)       --- pandas DataFrame containing the data to plot the histogram
    feat (str)         --- name of the feature to compute the histogram
    n_buckets (int)    --- number of bins of the histrogram
    scale (bool)       --- whether to scale the values so the sum of values equals to 1
    left_outliers (float) --- percentile threshold to consider outliers
    right_outliers (float) --- percentile threshold to consider outliers
    agg_params (dict)  --- dictionary containing the features as keys, and a list of aggregations as values
    
    Returns
    -----------------
    hist (DataFrame)   --- pandas DataFrame containing the bins, the histogram and the rest of aggregations
    """
    # list containing the features invoved in the calculations
    feats = [feat] 
    if agg_params is not None: feats = feats + [i for i in agg_params.keys()]
    hist = X[feats]
    
    # new dict containing the aggregations: count for 'feat' and the same agregations for
    # the rest of features
    agg_dict = {feat: 'count'} if hist[feat].dtype.name in ['category', 'object'] else {feat: ['count', 'max']}
    if agg_params is not None: agg_dict.update(agg_params)
    
    # chek if the X is of numerical or categorical value
    if hist[feat].dtype.name in ['category', 'object']:
        hist = hist.groupby([feat]).agg(agg_dict)
        hist.columns = list(map('_'.join, hist.columns.values))
        hist = hist.reset_index()
                
    else:
        hist.loc[:, 'bin'] = pd.cut(hist[feat], bins = n_buckets, labels = False, duplicates = 'drop') + 1

        if left_outliers is not None and right_outliers is not None:
            subset = np.logical_and(hist[feat] >= hist[feat].quantile(left_outliers),
                    hist[feat] <= hist[feat].quantile(right_outliers))

            hist.loc[subset, 'bin'] = pd.cut(hist.loc[subset, feat], bins = n_buckets, labels = False, duplicates = 'drop') + 1
            hist.loc[hist[feat] < hist[feat].quantile(left_outliers), 'bin'] = 0
            hist.loc[hist[feat] > hist[feat].quantile(right_outliers), 'bin'] = n_buckets + 1

        elif left_outliers is not None:
            subset = hist[feat] >= hist[feat].quantile(left_outliers)

            hist.loc[subset, 'bin'] = pd.cut(hist.loc[subset, feat], bins = n_buckets, labels = False, duplicates = 'drop') + 1
            hist.loc[hist[feat] < hist[feat].quantile(right_outliers), 'bin'] = 0

        elif right_outliers is not None:
            subset = hist[feat] <= hist[feat].quantile(right_outliers)

            hist.loc[subset, 'bin'] = pd.cut(hist.loc[subset, feat], bins = n_buckets, labels = False, duplicates = 'drop') + 1
            hist.loc[hist[feat] > hist[feat].quantile(right_outliers), 'bin'] = n_buckets + 1

        else:
            hist['bin'] = pd.cut(hist[feat], bins = n_buckets, labels = False, duplicates = 'drop') + 1

        hist = hist.groupby(['bin'], as_index = False).agg(agg_dict)
        hist.columns = list(map('_'.join, hist.columns.values))
        hist = hist.reset_index(drop = True)
    
    # if scale is true, divide count by total count
    if scale:
        hist[feat + '_count'] = hist[feat + '_count'] / hist[feat + '_count'].sum()
                
    return hist