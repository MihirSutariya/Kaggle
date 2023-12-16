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
        train_df['SalePrice'] = np.log(train_df['SalePrice'])
        train_df['GrLivArea'] = np.log(train_df['GrLivArea'])
        train_df.loc['TotalBsmtSF'] = np.log(train_df['TotalBsmtSF'])
        train_df.fillna(0,inplace=True)
        Nulls = ['PoolQC', 'MiscFeature','Alley','Fence','FireplaceQu','LotFrontage']
        train_df.drop(Nulls,axis=1,inplace=True)
        
        
    def train(self):
        y = train_df['SalePrice']
        X = train_df.drop(['SalePrice','Id'],axis=1)

    
        X_train, X_val, y_train, y_val =  train_test_split(X,y,test_size=0.01,random_state=42)
        self.rf_regressor = RandomForestRegressor(   n_estimators=100, 
                                        random_state=42,
                                        max_depth=20,
                                        min_samples_split=3,
                                        min_samples_leaf = 2,
                                        max_features = 30
                                            )
        self.rf_regressor.fit(X_train, y_train)
        
    def preprocess(self,df):
        for column, mapping in self.factorized_mappings.items():
            if column in df.columns and df[column].dtype == 'object':
                df[column] = df[column].map(mapping).fillna(-1)
        df.fillna(-1,inplace = True)
        df['GrLivArea'] = np.log(df['GrLivArea'])
        df.loc['TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
        df.fillna(0,inplace=True)
        Nulls = ['PoolQC', 'MiscFeature','Alley','Fence','FireplaceQu','LotFrontage']
        df.drop(Nulls,axis=1,inplace=True)
        self.test_df = df
        
    def predict(self):
        T = self.test_df.copy()
        T.drop('Id',axis = 1,inplace = True)
        predictions = self.rf_regressor.predict(T)
        self.results = pd.DataFrame()
        
        self.results['Id']  = self.test_df['Id'].astype('int')
        self.results['SalePrice'] = np.exp(predictions)
        
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