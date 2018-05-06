# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:03:15 2018

@author: monch
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score 

import imp
module_name = str(int(1000000000 * np.random.rand()))
problem = imp.load_source('', 'problem.py')

X_df, y_array = get_train_data()


#[T, X, Vx, Ax, Jx, Y, Vy, Ay, Jy, Z, Vz, Az, Jz, U2, C2, U3, C3, T3] =\
#    range(18)
[T, X, Vx, Ax, Jx, Y, Vy, Ay, Jy, Z, Vz, Az, Jz, U2, C2, U3, C3, T3, vp, va, vC2, vz, za, zj, d] =\
    range(25)  
############## Idea base#################################################################################
class FeatureExtractor():
    def __init__(self):
        pass

    # use this if you need to learn something at training time that depends on the labels
    # this will not be called on the test instances
    def fit(self, X_df, y):
        pass

    # this will be called both on the training and test instance
    def transform(self, X_df):
        data_matrix = np.asarray(list(X_df['data'].values))
        data_matrix[:,:,T] = 0
        data_matrix[:,:,X] = 0
        data_matrix[:,:,Y] = 0
        data_matrix[:,:,Vx] = 0
        data_matrix[:,:,U2] = 0
        #data_matrix[:,:,T3] = np.log(1e-10 + abs(data_matrix[:,:,T3]))
        #data_matrix[:,:,U3] = np.log(1e-10 + abs(data_matrix[:,:,U3]))  il donne la même accurancy 
        data_matrix[:,:,C2] = np.log(1e-10 + abs(data_matrix[:,:,C2]))
        data_matrix[:,:,C3] = np.log(1e-10 + abs(data_matrix[:,:,C3]))
        meadians_abs = abs(np.median(data_matrix,axis=1))
        
        return meadians_abs
    
####################################################################################################
# CLASSIFIERS
####################################################################################################
        
    
class Classifier(BaseEstimator): #RandomForest
    def __init__(self):
        pass

    def fit(self, X, y):
        self.param_grid = {'n_estimators': range(30,90),'max_leaf_nodes': range(50, 300)}
        self.clf_0 = RandomForestClassifier(random_state=61)
        self.randomized_mse = RandomizedSearchCV(estimator=self.clf_0, param_distributions=self.param_grid, n_iter=5,scoring="neg_log_loss", cv=5, verbose=1)
        print("5 folder cross validation with Random Forest Classifier : ")
        self.randomized_mse.fit(X,y)
        print("Best parameters found : ", self.randomized_mse.best_params_)
        print("Lowest log loss found : ", np.sqrt(np.abs(self.randomized_mse.best_score_)))
        self.clf = RandomForestClassifier(n_estimators=self.randomized_mse.best_params_['n_estimators'], max_leaf_nodes=self.randomized_mse.best_params_['max_leaf_nodes'], random_state=61)
        self.clf.fit(X,y)
    def predict_proba(self, X):
        return self.clf.predict_proba(X) 

class Classifier2(BaseEstimator): #XGBoost
    def __init__(self):
        pass

    def fit(self, X, y):
        
        self.param_grid = {'n_estimators': range(30,60),'max_depth': range(3,10),'learning_rate' : (0,1)}
        self.clf_0 = xgb.XGBClassifier(seed=61)
        self.randomized_mse = RandomizedSearchCV(estimator=self.clf_0, param_distributions=self.param_grid, n_iter=5,scoring="neg_log_loss", cv=5, verbose=1)
        print("5 folder cross validation with Xgboost Classifier : ")
        self.randomized_mse.fit(X,y)
        print("Best parameters found : ", self.randomized_mse.best_params_)
        print("Lowest log loss found : ", self.randomized_mse.best_score_)
        self.clf = xgb.XGBClassifier(n_estimators=self.randomized_mse.best_params_['n_estimators'], max_depth=self.randomized_mse.best_params_['max_depth'], learning_rate=self.randomized_mse.best_params_['learning_rate'])
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
#######################################################################################################

fe = FeatureExtractor()
fe2 = FeatureExtractor()
fe.fit(X_df, y_array)
fe2.fit(X_df,y_array)
X_array = fe.transform(X_df)
X2_array = fe2.transform(X_df)
print("-----------------------------------------------------")
print("X_array.shape : " + str(X_array.shape))
print("X2_array.shape : " + str(X2_array.shape))

clf = Classifier() #Random Forest
clf2 = Classifier2() #Xgboost

clf.fit(X_array, y_array)
clf2.fit(X2_array, y_array)

y_proba_array = clf.predict_proba(X_array)
y2_proba_array = clf2.predict_proba(X2_array)

print("-----------------------------------------------------")
print("y_proba_array.shape" + str(y_proba_array.shape))
print("y2_proba_array.shape" + str(y2_proba_array.shape))

y_pred_array = [problem._prediction_label_names[y] for y in np.argmax(y_proba_array, axis=1)]
y2_pred_array = [problem._prediction_label_names[y] for y in np.argmax(y2_proba_array, axis=1)]

print("-----------------------------------------------------")
print("Accurancy train base : " + str(accuracy_score(y_array, y_pred_array)))
print("Accurancy train new model : " + str(accuracy_score(y_array, y2_pred_array)))
print("diference train base-new model : " +  str(accuracy_score(y_array, y_pred_array)-accuracy_score(y_array, y2_pred_array)))

X_test_df, y_test_array = get_test_data()

X_test_array = fe.transform(X_test_df)
X2_test_array = fe2.transform(X_test_df)

y_test_proba_array = clf.predict_proba(X_test_array)
y2_test_proba_array = clf2.predict_proba(X2_test_array)

y_test_pred_array = [problem._prediction_label_names[y] for y in np.argmax(y_test_proba_array, axis=1)]
y2_test_pred_array = [problem._prediction_label_names[y] for y in np.argmax(y2_test_proba_array, axis=1)]

print("--------------------------------------------------------")
print("Accurancy test base : " + str(accuracy_score(y_test_array, y_test_pred_array)))
print("Accurancy test new model : " + str(accuracy_score(y_test_array, y2_test_pred_array)))
print("diferencia test base-new model : " +  str(accuracy_score(y_test_array, y_test_pred_array)-accuracy_score(y_test_array, y2_test_pred_array)))
print("----------------------------------------------------------")
            
