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

class GradeMaxTen(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 10 if x > 10 else x) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 10 if x > 10 else x) 
        
        return data

class GradeMinZero(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        data['NOTA_DE'] = data['NOTA_DE'].apply(lambda x: 0 if x < 0 else x) 
        data['NOTA_EM'] = data['NOTA_EM'].apply(lambda x: 0 if x < 0 else x) 
        data['NOTA_MF'] = data['NOTA_MF'].apply(lambda x: 0 if x < 0 else x) 
        data['NOTA_GO'] = data['NOTA_GO'].apply(lambda x: 0 if x < 0 else x) 
        
        return data

class MeanGradeIfNaN(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        for index, row in data.iterrows():
            if(type(row['NOTA_DE']) == float and pd.isna(row['NOTA_DE'])):
                data.loc[data.index == index, 'NOTA_DE'] = row['NOTA_EM']
            if(type(row['NOTA_EM']) == float and pd.isna(row['NOTA_EM'])):
                data.loc[data.index == index, 'NOTA_EM'] = row['NOTA_DE']
            if(type(row['NOTA_MF']) == float and pd.isna(row['NOTA_MF'])):
                data.loc[data.index == index, 'NOTA_MF'] = row['NOTA_GO']
            if(type(row['NOTA_GO']) == float and pd.isna(row['NOTA_GO'])):
                data.loc[data.index == index, 'NOTA_GO'] = row['NOTA_MF']
        
        return data

class MeanGrades(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        for index, row in data.iterrows():
            meanExact = round((row['NOTA_DE'] + row['NOTA_EM']) / 2, 1)
            menaHuman = round((row['NOTA_MF'] + row['NOTA_GO']) / 2, 1)
            data.loc[data.index == index, 'NOTA_DE'] = meanExact
            data.loc[data.index == index, 'NOTA_EM'] = meanExact
            data.loc[data.index == index, 'NOTA_MF'] = menaHuman
            data.loc[data.index == index, 'NOTA_GO'] = menaHuman
        
        return data
    
class UpdateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,features):
        self.features = features
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        features = ["NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO"]

        return features