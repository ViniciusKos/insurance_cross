

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
