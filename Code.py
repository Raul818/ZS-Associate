

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn import decomposition,pipeline
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble, preprocessing, grid_search, cross_validation
from sklearn import metrics 
from sklearn.calibration import CalibratedClassifierCV

import scipy.stats as scs
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer



if __name__ == '__main__':
    
    
    
    
    HospitalProfiling  = pd.read_csv('../Dataset/HospitalProfiling.csv')#, nrows =1000)
    HospitalRevenue  = pd.read_csv('../Dataset/HospitalRevenue.csv')
    ProjectedRevenue = pd.read_csv('../Dataset/ProjectedRevenue.csv')
    Solution = pd.read_csv('../Dataset/Solution.csv')
    
   
    
    
    
    
    
    
    District_ID = Solution.District_ID
    Hospital_ID = Solution.Hospital_ID
    Instrument_ID = Solution.Instrument_ID
    
    
    
    
    text_columns = []
    HospitalRevenue = HospitalRevenue.drop('Region_ID',axis = 1)
    for f in HospitalRevenue.columns:
        if HospitalRevenue[f].dtype=='object':
            print f,len(np.unique(HospitalRevenue[f].values))
            if f != 'loca':
                text_columns.append(f)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(HospitalRevenue[f].values) + list(Solution[f].values))
                HospitalRevenue[f] = lbl.transform(list(HospitalRevenue[f].values))
                Solution[f] = lbl.transform(list(Solution[f].values))   
    
    
    Hospital_ID_unique = np.unique(list(HospitalRevenue.Hospital_ID.values) + list(Solution.Hospital_ID.values))
    District_ID_unique = np.unique(list(HospitalRevenue.District_ID.values) + list(Solution.District_ID.values))
    Instrument_ID_unique = np.unique(list(HospitalRevenue.Instrument_ID.values) + list(Solution.Instrument_ID.values))
    
    
    add = []
    for i in Hospital_ID_unique:
        for j in District_ID_unique:
            for k in Instrument_ID_unique:
                add.append([i,j,k])
                
    Full = pd.DataFrame(add)
    
    Full.rename(columns = {0:'Hospital_ID',1:'District_ID',2:'Instrument_ID'},inplace = True)
    
    Full['Buy_or_not'] = -1
    
    
    Full = Full.merge(HospitalRevenue[[ 'Instrument_ID','District_ID','Hospital_ID','Year Total']],on = ['Hospital_ID','District_ID','Instrument_ID'], how = 'left')
                
    Full.replace(np.nan , -1,inplace = True)
    
    Full.loc[Full['Year Total'] == -1, 'Buy_or_not' ] = 0
    Full.loc[Full['Year Total'] != -1, 'Buy_or_not' ] = 1
    y = Full.Buy_or_not
    
    train = Full.drop(['Year Total','Buy_or_not'], axis =1)
    
    test = Solution.drop(['Buy_or_not','Revenue'],axis=1)
    
    
    
    


   
    print test.shape
    k_freq_instruments = HospitalRevenue[['District_ID', 'Instrument_ID']].groupby('Instrument_ID').agg('count').reset_index()
    k_freq_instruments.rename(columns = {'District_ID':'count_inst'},inplace = True)
    
    
    k_freq_hosp = HospitalRevenue[['Hospital_ID', 'Instrument_ID']].groupby('Hospital_ID').agg('count').reset_index()
    k_freq_hosp.rename(columns = {'Instrument_ID':'count_hosp'},inplace = True)
    
    k_freq_dist = HospitalRevenue[['Hospital_ID', 'District_ID']].groupby('District_ID').agg('count').reset_index()
    k_freq_dist.rename(columns = {'Hospital_ID':'count_district'},inplace = True)
    
    k_freq_inst_dist = HospitalRevenue[['Hospital_ID', 'Instrument_ID','District_ID']].groupby(['Instrument_ID','District_ID']).agg('count').reset_index()
    k_freq_inst_dist.rename(columns = {'Hospital_ID':'count_inst_district'},inplace = True)  
    
    train = train.merge(k_freq_dist , on = 'District_ID',how = 'left')
    train = train.merge(k_freq_hosp, on = 'Hospital_ID',how = 'left')
    train = train.merge(k_freq_instruments , on = 'Instrument_ID',how = 'left')
    train = train.merge(k_freq_inst_dist, on = ['Instrument_ID','District_ID'],how = 'left')
    
    test = test.merge(k_freq_dist , on = 'District_ID',how = 'left')
    test = test.merge(k_freq_hosp, on = 'Hospital_ID',how = 'left')
    test = test.merge(k_freq_instruments , on = 'Instrument_ID',how = 'left')
    test = test.merge(k_freq_inst_dist, on = ['Instrument_ID','District_ID'],how = 'left')

    print test.shape
 
    
    
    
    
    
    
    train.replace(np.nan, -1,inplace = True)
    test.replace(np.nan,-1,inplace = True)
    
    
   
    
    
    #gbm = ensemble.GradientBoostingClassifier(random_state=42)
    #params = [{'n_estimators': [100], 'min_samples_split': [5],'max_depth': [4] , 'max_features' : ['sqrt'], 'learning_rate':[0.1]}]    
    #clf = grid_search.GridSearchCV(gbm, params, verbose=1,n_jobs = -1)
    
    
    #clf = LogisticRegression(penalty = 'l1',  max_iter = 100 )
    

    # cross validation
    print("k-Fold RMSLE:")
    #cv_rmsle = cross_validation.cross_val_score(clf, train, y, scoring='f1')
    #print(cv_rmsle)
    #print("Mean: " + str(cv_rmsle.mean()))

    
    clf = ensemble.GradientBoostingClassifier(n_estimators = 100 , min_samples_split = 5, max_depth =4,  max_features= 'sqrt', learning_rate = 0.1, random_state = 42)
    clf.fit(train, y)
    Revenue1 = clf.predict_proba(test)[:,1]
    
    clf = ensemble.GradientBoostingClassifier(n_estimators = 100 , min_samples_split = 5, max_depth =6,  max_features= 'sqrt', learning_rate = 0.1, random_state = 42)
    clf.fit(train, y)
    Revenue2 = clf.predict_proba(test)[:,1]
    
    clf = ensemble.GradientBoostingClassifier(n_estimators = 100 , min_samples_split = 5, max_depth =4,  max_features= 'sqrt', learning_rate = 0.2, random_state = 42)
    clf.fit(train, y)
    Revenue3 = clf.predict_proba(test)[:,1]
    
    
    
    Revenue = Revenue1*0.2+ 0.4* Revenue2+ 0.4*Revenue3
    
    
    
  
    R1 = np.zeros(len(Revenue))
    R1[Revenue > .2] = 1
    

    lol = pd.DataFrame({"Hospital_ID":Hospital_ID,"District_ID":District_ID,"Instrument_ID":Instrument_ID,"Buy_or_not":R1,"Revenue": Revenue})
    lol[['Hospital_ID','District_ID','Instrument_ID','Buy_or_not','Revenue']].to_csv('D:/s21e.csv',index = False)
    
 
