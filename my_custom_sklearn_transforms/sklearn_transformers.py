from sklearn.base import BaseEstimator, TransformerMixin
import numpy
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class DataTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        # Replica the same grade to discipline of the same area
        for index, row in data.iterrows():
            if(type(row['NOTA_DE']) == float and pd.isna(row['NOTA_DE'])):
                data.loc[data.index == index, 'NOTA_DE'] = row['NOTA_EM']
            if(type(row['NOTA_EM']) == float and pd.isna(row['NOTA_EM'])):
                data.loc[data.index == index, 'NOTA_EM'] = row['NOTA_DE']
            if(type(row['NOTA_MF']) == float and pd.isna(row['NOTA_MF'])):
                data.loc[data.index == index, 'NOTA_MF'] = row['NOTA_GO']
            if(type(row['NOTA_GO']) == float and pd.isna(row['NOTA_GO'])):
                data.loc[data.index == index, 'NOTA_GO'] = row['NOTA_MF']        

        # Adjusting grades upper than 10
        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 10 if x > 10 else x) 

        # Adjusting grades lower than zero
        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 0 if x < 0 else x) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 0 if x < 0 else x) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 0 if x < 0 else x) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 0 if x < 0 else x) 

        # Sqrt grade
        # data['SQRT_NOTA_DE'] = data['NOTA_DE'].apply(lambda x: numpy.sqrt(x)) 
        # data['SQRT_NOTA_EM'] = data['NOTA_EM'].apply(lambda x: numpy.sqrt(x)) 
        # data['SQRT_NOTA_MF'] = data['NOTA_MF'].apply(lambda x: numpy.sqrt(x)) 
        # data['SQRT_NOTA_GO'] = data['NOTA_GO'].apply(lambda x: numpy.sqrt(x)) 

        # Bellow or Above the mean
        data['AUX_NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 0 if x < 7 else 1) 
        data['AUX_NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 0 if x < 7 else 1) 
        data['AUX_NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 0 if x < 7 else 1) 
        data['AUX_NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 0 if x < 7 else 1) 

        myColumns = [ #'SQRT_NOTA_DE', 'SQRT_NOTA_EM', 'SQRT_NOTA_MF', 'SQRT_NOTA_GO'
                     'AUX_NOTA_DE', 'AUX_NOTA_EM', 'AUX_NOTA_MF', 'AUX_NOTA_GO']


        return pd.DataFrame(data=data, index=None, columns=myColumns)
