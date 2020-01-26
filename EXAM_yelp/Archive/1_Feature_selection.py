#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:38:42 2019

@author: Anita Kurm and Maria Abildtrup Madsen
"""

###--------------------------------------###
###---------------SET UP-----------------###
###--------------------------------------###

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from textatistic import Textatistic #for readability metric
from collections import Counter
import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import json
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Remove warnings 
pd.set_option('mode.chained_assignment', None) # Remove warnings 

# Setting working directory 
os.chdir("/Users/mariaa.madsen/Google Drive/NLP Anita and Maria/Data")

#read data. 
df_total = pd.read_excel('yelp_subset.xlsx', sep=';', encoding='latin-1', nrows=10000)

#keep only relevant columns: stars, text, useful_dummy. 
df = df_total[['stars','text', 'useful_dummy']]

###--------------------------------------###
###-----------BASIC FEATURES-------------###
###--------------------------------------###

#num_char = len(text)

#number of words
def word_count(text):
    words = text.split()
    return len(words)

#average length of words
def avg_word_len(text):
    words = text.split()
    word_length = [len(word) for word in words]
    avg_word_len = sum(word_length)/len(words)
    return avg_word_len

#ADDING FEATURES TO DF
df['num_chars'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(word_count)
df['avg_word_len'] = df['text']. apply(avg_word_len)

###--------------------------------------###
###----------READABILITY FEATURES--------###
###--------------------------------------###

def text_readability(text):

    """Creates a Textatistic Object that contains various readability scores. Then extracts 2 of those scores: 
        1)Flesch reading ease
            greater average sentence length - harder to read; 
            greater avg num of syllables harder to read;
            higher the score - greater the readability (easier to understand)
            
        2)Gunning fog index
            Also utilizes average sentence length
            Greater % of complex words - harder to read
            higher the score - lesser the readability (harder to understand) """

    try:
        readability_scores = Textatistic(text).scores
        flesch = readability_scores['flesch_score']
        gunningfog = readability_scores['gunningfog_score']
        return flesch, gunningfog
    except:
        return np.nan, np.nan

#ADDING FEATURES TO DF
df['flesc'], df['gunningfog'] = zip(*df['text']. map(text_readability))

###--------------------------------------###
###-------------POS FEATURES-------------###
###--------------------------------------###

def count_pos(review):
    lower_case = review.lower()
    tokens = nltk.word_tokenize(lower_case)
    tags = nltk.pos_tag(tokens)
    countt = Counter( tag for word,  tag in tags)
    return countt

#get pos counts
df['pos'] = df['text']. apply(count_pos)

data_pos = pd.DataFrame(df['pos'].values.tolist(), index=df.index).fillna(0).astype(int)
df = df.join(data_pos)

#drop counter object, so the classifier works
df = df.drop(columns=['pos'])

###--------------------------------------###
###-------------REVIEW COUNT-------------###
###--------------------------------------###

### Rating deviance column ###
#Mapping the business rating values into the df 
df['review_count'] = df_total['review_count']

###--------------------------------------###
###-----------SENTIMENT FEATURES---------###
###-----------------Vader----------------###

#calling the sentiment analysis tool something shorter
analyzer = SentimentIntensityAnalyzer()


def get_vadersentiment(text):
    compound = analyzer.polarity_scores(text)['compound']
    pos = analyzer.polarity_scores(text)['pos']
    neu = analyzer.polarity_scores(text)['neu']
    neg = analyzer.polarity_scores(text)['neg']
    
    return compound, pos, neu, neg


df['sent_compound'], df['sent_pos'], df['sent_neu'], df['sent_neg'] = zip(*df['text']. map(get_vadersentiment))


###--------------------------------------###
###----------SENTIMENT ANALYIS-----------###
###--------------TextBlob----------------###

def detect_polarity(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df.text.apply(detect_polarity)

def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

df['subjectivity'] = df.text.apply(detect_subjectivity)



###--------------------------------------###
###------------RATING DEVIANCE-----------###
###--------------------------------------###

### Rating deviance column ###
#Mapping the business rating values into the df 
df['rating_deviance'] = df_total['rating_deviance']

###--------------------------------------###
###------------Concreteness-------------###
###--------------------------------------###

#load concreteness data
conc_database = pd.read_excel('Concreteness_ratings.xlsx', encoding='latin-1')

def get_concreteness(text):
    word_list = text.split(' ')
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    concreteness_score = df_with_concr['Conc.M'].mean()
    return concreteness_score

df['concreteness'] = df['text']. apply(get_concreteness)


###--------------------------------------###
###-----------MODEL EVALUATION-----------###
###--------------------------------------###

df = df.dropna()
df = df.drop('text', axis =1)
y = df['useful_dummy'].values
X = df.drop('useful_dummy', axis=1).values

def run_crossval_logreg(data):
    clf = LogisticRegression()
    #fit and get score of clf using cross validation
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision_score = scores['test_precision_macro'].mean()
    recall_score = scores['test_recall_macro'].mean()
    return precision_score, recall_score

crossval_logreg = run_crossval_logreg(df)

def run_crossval_knn(data):
    clf = KNeighborsClassifier(n_neighbors = 7)
    #fit and get score of clf using cross validation
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision_score = scores['test_precision_macro'].mean()
    recall_score = scores['test_recall_macro'].mean()
    return precision_score, recall_score

crossval_knn = run_crossval_knn(df)

###--------------------------------------###
###-----------FEATURE SELECTION----------###
###--------------------------------------###

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel


"""### LogReg ###
y = df['useful_dummy'].values
X = df.drop('useful_dummy', axis=1).values
# Fit the classifier to the training data
logreg_selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
logreg_selector.estimator_.coef_
logreg_selector.threshold_
logreg_selector.get_support()
X_new = logreg_selector.transform(X)

print(X_new)

"""
from sklearn import linear_model

### Lasso ###

# Fit the classifier to the training data
lasso_selector = SelectFromModel(estimator=linear_model.Lasso()).fit(X, y)
lasso_selector.estimator_.coef_
lasso_selector.threshold_
list_true = lasso_selector.get_support()
X_new = lasso_selector.transform(X)

df_features = df.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns)
# Try to select features with Lasso



#plot feature importance
# Plot the feature importances
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

###--------------------------------------###
###------------MODEL EVALUATION 2--------###
###--------------------------------------###

def run_crossval_logreg(data):
    clf = LogisticRegression()
    #fit and get score of clf using cross validation
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
    precision_score = scores['test_precision_macro'].mean()
    recall_score = scores['test_recall_macro'].mean()
    return precision_score, recall_score

crossval_logreg_new = run_crossval_logreg(df)

def run_crossval_knn(data):
    clf = KNeighborsClassifier(n_neighbors = 7)
    #fit and get score of clf using cross validation
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, X_new, y, scoring=scoring, cv = 5)
    precision_score = scores['test_precision_macro'].mean()
    recall_score = scores['test_recall_macro'].mean()
    return precision_score, recall_score



crossval_knn_new = run_crossval_knn(df)

print('Precision and recall on LogReg model: ' + str(crossval_logreg))
print('Precision and recall on KNN model: ' + str(crossval_knn))
print('Precision and recall on LogReg model after feature selection: ' + str(crossval_logreg_new))
print('Precision and recall on KNN model after feature selection: ' + str(crossval_knn_new))

"""
##### Results when running it on 10.000 reviews: #####

Precision and recall on LogReg model: (0.646035270023598, 0.5183310211166022)
Precision and recall on KNN model: (0.5728703640770552, 0.5440257583908734)
Precision and recall on LogReg model after feature selection: (0.6511730389945457, 0.5184150768810085)
Precision and recall on KNN model after feature selection: (0.5737870980665627, 0.5410240946085842)
"""