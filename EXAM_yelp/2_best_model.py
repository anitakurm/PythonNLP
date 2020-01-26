#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:13:33 2019

@author: Maria Abildtrup Madsen and Anita Kurm
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
import os
import warnings
from matplotlib import pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning) # Remove warnings 
pd.set_option('mode.chained_assignment', None) # Remove warnings 

# Setting working directory 
os.chdir("/Users/mariaa.madsen/Google Drive/NLP Anita and Maria/Data")

#read data. 
df = pd.read_excel('yelp_subset_features.xlsx', sep=';', encoding='latin-1', nrows=10000)    #subset where not useful<2 and useful >1
#df = pd.read_excel('yelp_subset_features2.xlsx', sep=';', encoding='latin-1', nrows=10000)  #subset where not useful=0 and useful >2

#%%

###--------------------------------------###
###-----------FEATURE SELECTION----------###
###--------------------------------------###

# importing necessary modules 
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Preparing data
data = df.dropna()
data = data.drop('text', axis =1)
y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values

#%%

""" TO DO: CHOOSE ONE OF THE BELOW METHODS FOR MODEL SELECTION """
chosen_method = LogisticRegression
#chosen_method = LinearSVC

""" Running the feature selection on the chosen method """
method = chosen_method(C=0.01, penalty="l1", dual=False).fit(X, y)
model=SelectFromModel(method, prefit=True)
list_true = model.get_support()
X_new = model.transform(X)

# Returns a dataframe with chosen features     
df_features = data.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns)

#%%
# Calulating correlations 
cor = df.corr()
cor_target = abs(cor["useful_dummy"])
correlations = cor_target
#correlations = cor_target[cor_target>0.3]
print(correlations)

#%%
correlations = pd.DataFrame(correlations)
df.to_excel('correlations.xlsx')

#%%

###--------------------------------------###
###------------CROSS VALIDATION ---------###
###--------------------------------------###

""" LOGISTIC REGRESSION """
clf = LogisticRegression()
scoring = ['precision_macro', 'recall_macro', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
precision = scores['test_precision_macro'].mean() 
recall = scores['test_recall_macro'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_logreg = precision, recall, accuracy_balanced, f1_weighted
    
""" K NEAREST NEIGHBORS"""
clf = KNeighborsClassifier(n_neighbors = 7)
scoring = ['precision_macro', 'recall_macro', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
precision = scores['test_precision_macro'].mean() 
recall = scores['test_recall_macro'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_knn = precision, recall, accuracy_balanced, f1_weighted

""" SVC """
clf = SVC(gamma='auto')
scoring = ['precision_macro', 'recall_macro', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
precision = scores['test_precision_macro'].mean() 
recall = scores['test_recall_macro'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_svc = precision, recall, accuracy_balanced, f1_weighted

""" RANDOM FORREST """
clf = RandomForestClassifier(max_depth=2, random_state=0)
scoring = ['precision_macro', 'recall_macro', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
precision = scores['test_precision_macro'].mean() 
recall = scores['test_recall_macro'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_randomforrest = precision, recall, accuracy_balanced, f1_weighted