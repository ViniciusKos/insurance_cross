import pickle
import pandas as pd
from flask import Flask, request, Response
from insurance import Insurance
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import data_cleaning, FeatureEngineering, BayesianEncoding

class data_cleaning(BaseEstimator, TransformerMixin,object):
    def __init__(self):
        pass
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        X2=X.copy()
        for i in ['region_code','policy_sales_channel']:
            X[i]=X[i].astype('object')    
        return X2

class FeatureEngineering(BaseEstimator, TransformerMixin, object):
    def __init__(self):
        pass
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        X2=X.copy()
        X2.columns=X2.columns.str.lower()
        X2['vehicle_age']=X2['vehicle_age'].map({'> 2 Years':'2. 2_more_years','1-2 Year':'1. 1_2_years', '< 1 Year': '0. 1_less_years'})
        return X2

class BayesianEncoding(BaseEstimator, TransformerMixin, object):
    def __init__(self):
        pass
    def fit(self, X,y=None):
        self.dic={}
        X2=pd.concat([X,y],axis=1)
        for i in X.columns:
            self.dic[i]=X2.groupby(i)['response'].mean()/X2['response'].mean()
        return self
    def transform(self, X, y=None):
        X2=X.copy()
        for i in self.dic.keys():
            X2[f'{i}']=X2[i].map(self.dic[i])
        return X2


# loading model
model = pickle.load( open(r'parameters/model.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( r'/predict', methods=['POST'] )
def insurance_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Insurance class
        pipeline = Insurance()
        
        # data cleaning
        #df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engineering( test_raw )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
        
        
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( '' ,debug=True)