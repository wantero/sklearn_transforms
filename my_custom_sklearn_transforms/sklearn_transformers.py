from sklearn.base import BaseEstimator, TransformerMixin


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