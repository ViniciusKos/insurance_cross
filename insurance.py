import pickle
import pandas as pd
import numpy as np



class Insurance( object ):
    def __init__( self ):
        self.home_path='P:\Python\GitHub\insurance_cross'
        self.cleaner = pickle.load( open( self.home_path + '\parameters\data_cleaning.pkl', 'rb') )
        self.fe = pickle.load( open( self.home_path + '\parameters\feature_engineering.pkl', 'rb') )
        self.oe = pickle.load( open( self.home_path + '\parameters\oe.pkl', 'rb') )
        self.be = pickle.load( open( self.home_path + '\parameters\bayesian_encoder.pkl', 'rb') ) 
        self.model = pickle.load( open( self.home_path + '\parameters\model.pkl', 'rb') )
        self.rescaler = pickle.load( open( self.home_path + '\parameters\rescaler.pkl', 'rb') )


        
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


