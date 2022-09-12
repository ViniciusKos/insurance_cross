import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from transformers import data_cleaning, FeatureEngineering, BayesianEncoding

class Insurance( object ):
    def __init__( self ):


        self.cleaner = pickle.load( open( r'parameters/data_cleaning.pkl', 'rb') )
        self.fe = pickle.load( open( r'parameters/feature_engineering.pkl', 'rb') )
        self.oe=pickle.load( open(r'parameters/oe.pkl', 'rb') )
        self.be = pickle.load( open( r'parameters/bayesian_encoder.pkl', 'rb') ) 
        self.robust=pickle.load(open(r"parameters/RobustScaler.pkl",'rb'))
        self.minmax=pickle.load(open(r"parameters/MinMaxScaler.pkl",'rb'))
        self.standard=pickle.load(open(r"parameters/StandardScaler.pkl",'rb'))


        
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
        df3[['annual_premium','policy_sales_channel']]=self.robust.transform(df3[['annual_premium','policy_sales_channel']])
        df3[['region_code','age']]=self.minmax.transform(df3[['region_code','age']])
        df3[['vintage']]=self.standard.transform(df3[['vintage']])
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


