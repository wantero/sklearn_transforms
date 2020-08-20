from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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

        # Adjusting grades upper than 10
        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 10 if x > 10 else x) 

        # Adjusting grades lower than zero
        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 0 if x < 0 or np.isnan(x) else x) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 0 if x < 0 or np.isnan(x) else x) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 0 if x < 0 or np.isnan(x) else x) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 0 if x < 0 or np.isnan(x) else x) 

        # Bellow or Above the mean
        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 0 if x < 7 else 1) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 0 if x < 7 else 1) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 0 if x < 7 else 1) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 0 if x < 7 else 1) 

        # Has disapproved?
        data['REPROVACOES_DE'] = data['REPROVACOES_DE'].apply(lambda x: 1 if x > 1 else 0) 
        data['REPROVACOES_EM'] = data['REPROVACOES_EM'].apply(lambda x: 1 if x > 1 else 0) 
        data['REPROVACOES_MF'] = data['REPROVACOES_MF'].apply(lambda x: 1 if x > 1 else 0) 
        data['REPROVACOES_GO'] = data['REPROVACOES_GO'].apply(lambda x: 1 if x > 1 else 0)         

        return data
