import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from feature_engineering.histogram import histogram


def plot_histogram(X, feat, n_buckets = 10, round_x = 2):

    # set the figure
    _, ax1 = plt.subplots(figsize = (11, 7))

    # remove top and right axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # compute the histogram values
    hist = histogram(X, feat, n_buckets = n_buckets, scale = True)

    # plot the figure
    # TODO: agregar la posibilidad de pintar valores en el eje secundario
    if X[feat].dtype.name in ['category', 'object']:
        ax1.bar(hist[feat], hist[feat + '_count'], edgecolor = 'darkgray', color = 'lightgray')
    else:
        ax1.bar(hist['bin_'], hist[feat + '_count'], edgecolor = 'darkgray', color = 'lightgray')

    # set the labels of the x-axis to the max of the category
    plt.xticks(ticks = hist['bin_'], labels = hist[feat + '_max'].apply(lambda x: str(np.round(x, round_x))))

    # label the axis
    ax1.set_xlabel(feat)
    ax1.set_ylabel('frequency of ' + feat)

    # return the data frame containing the values of the histogram
    return hist