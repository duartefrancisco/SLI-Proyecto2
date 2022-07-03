import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class CategoricalEncoderOperator(TransformerMixin, BaseEstimator):

    def __init__(self, columnas):
        self.columnas = columnas
    
    def fit(self, X, y=None):
        self.mapper = {}
        for columna in self.columnas:
            self.mapper[columna] = X[columna].value_counts().to_dict()
        return self

    def transform(self, X, y = None):
        X = X.copy()
        for columna in self.columnas:
            X[columna] = X[columna].map(self.mapper[columna])
        return X

class CategoricalCCAEnconderOperator(TransformerMixin, BaseEstimator):

    def __init__(self, columna):
        self.columna = columna
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        X = X.dropna()

        return X

class TargetEncoderOperator(TransformerMixin, BaseEstimator):

    def __init__(self, target):
        self.target = target

    def  fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        mapper = y.value_counts().to_dict()
        y = y.map(mapper)
        return y

class OutliersTreatmentOperator(BaseEstimator, TransformerMixin):

    def __init__(self, factor = 1.75, columnas = None):
        self.columnas = columnas
        self.factor = factor

    def fit(self, X, y = None):
        for col in self.columnas:
            q3 = X[col].quantile(0.75)
            q1 = X[col].quantile(0.25)
            self.IQR = q3 - q1
            self.upper = q3 + self.factor*self.IQR
            self.lower = q1 - self.factor*self.IQR
        return self

    def transform(self, X, y = None):
        X = X.copy()
        for col in self.columnas:
            X[col] = np.where(X[col] >= self.upper, self.upper,
                np.where(
                    X[col] < self.lower, self.lower, X[col]
                )    
            )
        return X