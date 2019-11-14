import pandas as pd

def import_dataset(dataset):

    if dataset == 'california_housing':
        data = pd.read_csv("./datasets/california_housing/california_housing.zip")

    return data
