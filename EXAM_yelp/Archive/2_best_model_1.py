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
df = pd.read_excel('yelp_subset_features_2.xlsx', sep=';', encoding='latin-1', nrows=10000)

#%%

###--------------------------------------###
###--------------EVALTUATING-------------###
###--------------------------------------###


#classifier performed using cross-validation (different scores)
#scores info can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
#and here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html 
#precision macro -  the number of true positives () over the number of true positives plus the number of false positives 
#recall macro - the ability of the classifier to find all the positive samples.

# importing necessary modules 
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import recall_score
from sklearn import linear_model

### Feature Selection ###

df = df.dropna()
#df = df.drop('text', axis =1)
y = df['useful_dummy'].values
X = df.drop('useful_dummy', axis=1).values

### LogReg
## Fit the classifier to the training data
#lasso_selector = SelectFromModel(LogisticRegression(C=0.01, penalty='l1')).fit(X, y)
#list_coef = lasso_selector.estimator_.coef_
#list_thres = lasso_selector.threshold_
#list_true = lasso_selector.get_support()
#print(np.sum(lasso_selector.estimator_.coef_ == 0))
#X_new = lasso_selector.transform(X)


### SVC
from sklearn.svm import LinearSVC
# Fit the classifier to the training data
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model=SelectFromModel(lsvc, prefit=True)
#list_coef = lasso_selector.estimator_.coef_
#list_thres = lasso_selector.threshold_
list_true = model.get_support()
#print(np.sum(lasso_selector.estimator_.coef_ == 0))
X_new = model.transform(X)


df_features = df.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns)

#%%

# plot feature importance

"""
importances = model.feature_importances_
importances['random_forest'] = rf.feature_importances_
criteria = criteria + ('random_forest',)
idx = 1

fig = plt.figure(figsize=(20, 10))
labels = ['$x_{}$'.format(i) for i in range(n)]
for crit in criteria:
    plt.subplot(2, 2, idx)
    plt.barrrr(numpy.arange(len(labels)),
            importances[crit],
            align='center',
            color='red')
    plt.xticks(numpy.arange(len(labels)), labels)
    plt.title(crit)
    plt.ylabel('importances')
    idx += 1
title = '$x_0,...x_9 \sim \mathcal{N}(0, 1)$\n$y= 10sin(\pi x_{0}x_{1}) + 20(x_2 - 0.5)^2 + 10x_3 + 5x_4 + Unif(0, 1)$'
fig.suptitle(title, fontsize="x-large")
plt.show()
"""

####### testing #########

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["useful_dummy"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.3]
relevant_features

##########################

#%%

###--------------------------------------###
###------------CROSS VALIDATION ---------###
###--------------------------------------###

# On logistic model


clf = LogisticRegression()
#fit and get score of clf using cross validation
scoring = ['accuracy', 'f1', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
accuracy = scores['test_accuracy'].mean() 
f1 = scores['test_f1'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_logreg = accuracy, f1, accuracy_balanced, f1_weighted


#%%

# On KNN model 

clf = KNeighborsClassifier(n_neighbors = 7)
#fit and get score of clf using cross validation
scoring = ['accuracy', 'f1', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
accuracy = scores['test_accuracy'].mean() 
f1 = scores['test_f1'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_knn = accuracy, f1, accuracy_balanced, f1_weighted

#%%

# On SVM model 

from sklearn.svm import SVC

clf = SVC(gamma='auto')
#fit and get score of clf using cross validation
scoring = ['accuracy', 'f1', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
accuracy = scores['test_accuracy'].mean() 
f1 = scores['test_f1'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_svc = accuracy, f1, accuracy_balanced, f1_weighted

#%%

# On RandomForrest model 
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
#fit and get score of clf using cross validation
scoring = ['accuracy', 'f1', 'balanced_accuracy', 'f1_weighted']
scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
accuracy = scores['test_accuracy'].mean() 
f1 = scores['test_f1'].mean() 
accuracy_balanced = scores['test_balanced_accuracy'].mean() 
f1_weighted = scores['test_f1_weighted'].mean() 
results_RandomForrest = accuracy, f1, accuracy_balanced, f1_weighted