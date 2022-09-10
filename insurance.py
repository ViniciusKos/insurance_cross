import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

class data_cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        X2=X.copy()
        for i in ['region_code','policy_sales_channel']:
            X[i]=X[i].astype('object')    
        return X2

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        X2=X.copy()
        X2.columns=X2.columns.str.lower()
        X2['vehicle_age']=X2['vehicle_age'].map({'> 2 Years':'2. 2_more_years','1-2 Year':'1. 1_2_years', '< 1 Year': '0. 1_less_years'})
        return X2

class BayesianEncoding(BaseEstimator, TransformerMixin):
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

class Rescaler(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.cols_robust=['annual_premium','policy_sales_channel']
        self.cols_minmax=['region_code','age']
        self.cols_standard=['vintage']
        pass 

    def fit(self, X,y=None):
        rs = RobustScaler()
        rs.fit( X[self.cols_robust].values )
        pickle.dump( rs, open( f'parameters/{rs.__class__.__name__}.pkl', 'wb') )

        mms=MinMaxScaler()
        mms.fit( X[self.cols_minmax].values )
        pickle.dump( mms, open( f'parameters/{mms.__class__.__name__}.pkl', 'wb') )

        ss=StandardScaler()
        ss.fit( X[self.cols_standard].values )
        pickle.dump( ss, open( f'parameters/{ss.__class__.__name__}.pkl', 'wb') )

        return self

    def transform(self, X, y=None):
        X2=X.copy()
        rs=pickle.load(open(f"parameters/RobustScaler.pkl",'rb'))
        mms=pickle.load(open(f"parameters/MinMaxScaler.pkl",'rb'))
        ss=pickle.load(open(f"parameters/StandardScaler.pkl",'rb'))

        X2[self.cols_robust]=rs.transform(X2[self.cols_robust])
        X2[self.cols_minmax]=mms.transform(X2[self.cols_minmax])
        X2[self.cols_standard]=ss.transform(X2[self.cols_standard])


        return X2

class Insurance( object ):
    def __init__( self ):
        self.cleaner = pickle.load( open( r'parameters/data_cleaning.pkl', 'rb') )
        self.fe = pickle.load( open( r'parameters/feature_engineering.pkl', 'rb') )
        self.oe=pickle.load( open(r'parameters/oe.pkl', 'rb') )
        self.be = pickle.load( open( r'parameters/bayesian_encoder.pkl', 'rb') ) 
        self.rescaler = pickle.load( open( r'parameters/rescaler.pkl', 'rb') )


        
    def data_cleaning( self, df1 ): 
        
        df1.columns=df1.columns.str.lower()
        df1=self.cleaner.transform(df1)

        return df1

    def feature_engineering( self, df2):

        df2=self.fe.transform(df2)

        return df2

        
    def data_preparation( self, df3 ):

        df3[['vehicle_age','gender','vehicle_damage']]=self.oe.transform(df3[['vehicle_age','gender','vehicle_damage']])
        df3[['policy_sales_channel','region_code']]=self.be.transform(df3[['policy_sales_channel','region_code']])
        df3=self.rescaler.transform(df3)
        df3=df3.fillna(0)
        df3=df3[['age', 'region_code', 'previously_insured', 'vehicle_age',
       'vehicle_damage', 'policy_sales_channel']]

        return df3
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = pred
        
        #return original_data
        return original_data.to_json( orient='records', date_format='iso' )


