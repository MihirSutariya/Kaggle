import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



class PriceModel:
    
    def __init__(self, train_df):
        self.train_df = train_df
        self.factorized_mappings = None
        self.test_df = None
        self.rf_regressor = None
        self.results = None
        self._prepro()
        
    def _prepro(self):
        non_numeric_columns = train_df.select_dtypes(exclude=['number']).columns.tolist()
        len(non_numeric_columns)
        self.factorized_mappings = {}
        # column_name : {'category': int}

        for column in train_df.columns:
            if train_df[column].dtype == 'object':  
                factorized_series, unique_values = pd.factorize(train_df[column], sort=True)
                self.factorized_mappings[column] = dict(zip(unique_values, [i for i in range(len(unique_values))]))
                train_df[column] = factorized_series
        
        train_df.fillna(-1,inplace=True)
        
    def train(self):
        y = train_df['SalePrice']
        X = train_df.drop(['SalePrice','Id'],axis=1)

    

        self.rf_regressor = RandomForestRegressor(   n_estimators=100,     # You can change the number of trees
                                                max_depth=20,         # Adjust the maximum depth of trees
                                                min_samples_leaf=5,   # Adjust the minimum samples per leaf
                                                random_state=42
                                            )
        self.rf_regressor.fit(X, y)
        
    def preprocess(self,df):
        for column, mapping in self.factorized_mappings.items():
            if column in df.columns and df[column].dtype == 'object':
                df[column] = df[column].map(mapping).fillna(-1)
        df.fillna(-1,inplace = True)
        self.test_df = df
        
    def predict(self):
        T = self.test_df.copy()
        T.drop('Id',axis = 1,inplace = True)
        predictions = self.rf_regressor.predict(T)
        self.results = pd.DataFrame()
        
        self.results['Id']  = self.test_df['Id']
        self.results['SalePrice'] = predictions
        
    def write_result(self):
        self.results.to_csv('./data/result.csv',index=False)
        
if __name__=="__main__":
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    train_df = pd.read_csv(train_path)
    p_model = PriceModel(train_df)
    test_df = pd.read_csv(test_path)
    p_model.train()
    p_model.preprocess(test_df)
    p_model.predict()
    p_model.write_result()
    
    
                