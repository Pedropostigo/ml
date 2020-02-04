"""
Module to compute the WoE (Weight of Evidence) for both numerical and categorical features
"""
import numpy as np
import pandas as pd

# TODO: arreglar estas funciones para que sean más óptimas y legibles
# TODO: guardar la tabla WoE en un data frame para luego transformar las variables
# TODO: crear una función para aplicar el WoE a nuevas variables

def woe_num(icol, binary_col, df, n_buckets):
    # Bucketizamos sobre todo el dataset
    df['bucket'], bins = pd.qcut(df[icol], q = n_buckets, labels = False, duplicates = 'drop',retbins=True)
    real_bins = len(bins) - 1

    # Creamos tabla WoE
    tabla_woe = df[['bucket', binary_col]].groupby(['bucket']).sum(skipna = True).reset_index()

    # Si algun bucket no tiene morosos, bucketizamos con uno menos:
    while 0 in tabla_woe[binary_col].values:
        df['bucket'], bins = pd.qcut(df[icol], q = real_bins - 1, labels = False, duplicates = 'drop', retbins = True)
        real_bins = len(bins) - 1
    
        # Creamos tabla WoE
        tabla_woe=df[['bucket', binary_col]].groupby(['bucket']).sum(skipna=True).reset_index()

    # Buenos y malos totales
    BAD = df[binary_col].sum(skipna=True)
    GOOD = df.loc[~df[binary_col].isna(),binary_col].count() - BAD

    # Nos aseguramos que al tirar los cortes repetidos con "duplicates='drop'" (esto pasa cuando
    # una variable tiene acumulación de valores repetidos) almenos queden 5 buckets restantes.
    # Si no, no woeizamos la variable.
    if real_bins >= 5:
        tabla_woe = tabla_woe.rename(columns={binary_col: 'bad'}) # Defaults
        tabla_woe['total']=df[['bucket',binary_col]].groupby(['bucket']).count().reset_index()[binary_col] # Totales
        tabla_woe['good']=(tabla_woe['total']-tabla_woe['bad']).astype(int) # Buenos

        # Cálculo WOE por bucket
        tabla_woe['woe']=np.log((tabla_woe['good'] / GOOD) / (tabla_woe['bad'] / BAD))
    
        # Unimos nuevo factor woeizado a la tabla y eliminamos factor original
        df=pd.merge(df, tabla_woe[['bucket','woe']], on = 'bucket', how = 'left')
        df = df.rename(columns = {'woe': icol + "_w"})
        df = df.drop(icol, axis = 1)
        df = df.drop('bucket', axis = 1)
    else:
        df = df.drop(icol, axis = 1)
        df = df.drop('bucket', axis = 1)

    return(df)


def woe_cat(icol, binary_col, df):

    """
    Function to compute the Weight of Evidence of a categorical feature

    Params:
    TODO: descripción de los parámetros que toma la función
    """

    # Creamos tabla WoE
    tabla_woe = df[[icol, binary_col]].groupby([icol]).sum(skipna = True).reset_index()
    real_bins = len(df[icol].unique()) - 1

    # Si algun bucket no tiene morosos, bucketizamos con uno menos:
    # while 0 in tabla_woe[binary_col].values:
    #    df['bucket'], bins = pd.qcut(df[icol], q = real_bins - 1, labels = False, duplicates = 'drop', retbins = True)
    #    real_bins = len(bins) - 1
    
        # Creamos tabla WoE
    #    tabla_woe=df[['bucket',binary_col]].groupby(['bucket']).sum(skipna=True).reset_index()
    
    # Buenos y malos totales
    BAD = df[binary_col].sum(skipna = True)
    GOOD = df.loc[~df[binary_col].isna(),binary_col].count() - BAD

    # Nos aseguramos que al tirar los cortes repetidos con "duplicates='drop'" (esto pasa cuando
    # una variable tiene acumulación de valores repetidos) almenos queden 5 buckets restantes.
    # Si no, no woeizamos la variable.
    if real_bins >= 5:
        tabla_woe = tabla_woe.rename(columns = {binary_col: 'bad'}) # Defaults
        tabla_woe['total'] = df[[icol, binary_col]].groupby([icol]).count().reset_index()[binary_col] # Totales
        tabla_woe['good'] = (tabla_woe['total'] - tabla_woe['bad']).astype(int) # Buenos

        # Cálculo WOE por bucket
        tabla_woe['woe'] = np.log((tabla_woe['good'] / GOOD) / (tabla_woe['bad'] / BAD))
        tabla_woe.loc[tabla_woe['woe'].isna(), 'woe'] = tabla_woe['woe'].max()
        
        # Unimos nuevo factor woeizado a la tabla y eliminamos factor original
        df=pd.merge(df, tabla_woe[[icol, 'woe']], on= icol, how = 'left')
        df = df.rename(columns={'woe': icol + "_w"})
        df = df.drop(icol, axis = 1)
    else:
        df = df.drop(icol, axis = 1)

    return(df)