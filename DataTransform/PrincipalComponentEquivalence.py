import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCE:
    def __init__(self, n=1):
        self.n = n
        self.X_mean = np.nan
        self.A = np.nan

    def split_list_randomly(self, input_list):
        # リストをシャッフル
        random.shuffle(input_list)
        # リストの長さを取得
        length = len(input_list)
        # 半分の長さを計算
        half_length = length // 2
        # リストを二等分
        first_half = input_list[:half_length]
        second_half = input_list[half_length:]
        return first_half, second_half
    
    def calc_D_inverse(self, V, X, X_mean):
        return (V.T@(X-X_mean).T) @ (V.T@(X-X_mean).T).T / (len(X)-1)

    def calc_A_step(self, X:pd.DataFrame):
        # X shape (n_samples, n_features)
        index = list(X.index)
        index1, index2 = self.split_list_randomly(index)
        X1 = X.loc[index1].astype(np.float64).values
        X2 = X.loc[index2].astype(np.float64).values
        pca = PCA()
        pca.fit(X1)
        V1 = pca.components_.T # V:eigenvector v V=(v1, v2, ..., vn)
        X1_mean = pca.mean_
        pca.fit(X2)
        V2 = pca.components_.T
        X2_mean = pca.mean_
        X_mean = (X1_mean+X2_mean) / 2
        D1_tilde_square = self.calc_D_inverse(V=V1, X=X2, X_mean=X_mean)
        D1_tilde_inverse = np.zeros(D1_tilde_square.shape)
        for i in range(len(D1_tilde_square)):
            D1_tilde_inverse[i,i] = 1 / np.sqrt(D1_tilde_square[i,i])
        D2_tilde_square = self.calc_D_inverse(V=V2, X=X1, X_mean=X_mean)
        D2_tilde_inverse = np.zeros(D2_tilde_square.shape)
        for i in range(len(D2_tilde_square)):
            D2_tilde_inverse[i,i] = 1 / np.sqrt(D2_tilde_square[i,i])
        A1 = V1 @ D1_tilde_inverse @ V1.T
        A2 = V2 @ D2_tilde_inverse @ V2.T
        self.X_mean = X_mean
        return (A1+A2)/2
    
    def fit(self, X:pd.DataFrame):
        # X shape (n_samples, n_features)
        _, N = X.shape
        A = np.zeros((N,N))
        for i in range(self.n):
            A += self.calc_A_step(X=X)
        self.A = A
        return True
        
    def transform(self, X:pd.DataFrame):
        Y = (self.A@X.T).T
        Y.columns = X.columns
        return Y
    
    def fit_transform(self, X:pd.DataFrame):
        self.fit(X=X)
        return self.transform(X=X)