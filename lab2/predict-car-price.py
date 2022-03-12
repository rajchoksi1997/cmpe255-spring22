import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import math

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
        
    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X]) 

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y) 
    
        return w[0], w[1:] 
        

    def prepare_X(self, input_data, base): 
        df_num = input_data[base]                   
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

def test() -> None:
    carPrice = CarPrice()
    carPrice.trim() 
    df = carPrice.df

    np.random.seed(2) 
    n = len(df) 
    n_val = int(0.2 * n) 
    n_test = int(0.2 * n) 
    n_train = n - (n_val + n_test)

    idx = np.arange(n) 
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx]
    
    
    df_train = df_shuffled.iloc[:n_train].copy() 
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()

   
    y_train_orig = df_train.msrp.values
    y_val_orig = df_val.msrp.values
    y_test_orig = df_test.msrp.values

    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)

    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity'] 
    X_train = carPrice.prepare_X(df_train, base) 

    w_0, w = carPrice.linear_regression(X_train, y_train)

    X_val = carPrice.prepare_X(df_val, base)
    y_pred_val = w_0 + X_val.dot(w)
    print("The rmse value of predicted MSRP and actual MSRP of validation set is ", carPrice.validate(y_val, y_pred_val))

    X_test = carPrice.prepare_X(df_test, base)
    y_pred_test = w_0 + X_test.dot(w)
    print("The rmse value of predicted MSRP and actual MSRP of test set is ", carPrice.validate(y_test, y_pred_test))


    y_pred_MSRP_val = np.expm1(y_pred_val) 
    
    df_val['msrp_pred'] = y_pred_MSRP_val 
    
    print("Let us print out first 5 cars in our Validation Set's original msrp vs. predicted msrp")
    print(df_val.iloc[:,5:].head(), "\n")

    with open("output.txt","w") as f:
        f.write(df_val.iloc[:,5:].head().to_string())
  


if __name__ == "__main__":
    test()