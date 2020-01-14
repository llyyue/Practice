# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:58:54 2020

@author: llyyue
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from sklearn.model_selection import KFold # import KFold


class Airbnb_dnn:
    
    def load_data(self):
        self.raw = pd.read_csv('.\listings.csv')
        p= self.raw['price']
        self.raw=self.raw[(p != 0) & (p< np.quantile(p, 0.99))]
        self.raw = self.raw.drop(['id','host_id','name','host_name'], axis=1)
        self.raw =self.raw.fillna(0)
        self.raw = pd.get_dummies(self.raw, columns=['neighbourhood_group','neighbourhood','room_type'])
        self.minmax_sc_x = MinMaxScaler()
        self.minmax_sc_y = MinMaxScaler()
        self.X = self.minmax_sc_x.fit_transform(self.raw.drop('price', axis=1))
        self.y = self.minmax_sc_y.fit_transform(self.raw['price'])
        print('')
    
    def build_model(self, X_train):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=X_train.shape[1], activation='tanh'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        
    def train(self,X_train, y_train,X_test, y_test):
        self.result = self.model.fit(X_train, y_train,
                          batch_size=64,
                          epochs=500,
                          validation_data=(X_test, y_test))
        self.score = self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def run_kfold(self, X, y):
        n_splits = 10
        kf = KFold(n_splits,random_state=None, shuffle=False) # Define the split - into 2 folds 
        
        rsArr=[]
        errArr=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.build_model(X_train)
            self.train(X_train, y_train,X_test, y_test)
            y_est = self.predict(X_test)
            
            err = np.abs(y_est - y_test)
            mse = np.sqrt(np.mean(err**2))
            mae = np.mean(np.mean(err))
            errArr.append([mse,mae])
        return errArr
            
if __name__ == '__main__':
    dnn = Airbnb_dnn()
    dnn.load_data()
    errArr = run_kfold()
    