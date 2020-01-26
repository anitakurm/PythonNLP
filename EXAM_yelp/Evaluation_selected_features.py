d#!/usr/bin/env python3
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore", category=FutureWarning) # Remove warnings 
pd.set_option('mode.chained_assignment', None) # Remove warnings 

# Setting working directory 
os.chdir("/Users/mariaa.madsen/Google Drive/NLP Anita and Maria/Data")

#read data. 
df_total = pd.read_excel('yelp_subset.xlsx', sep=';', encoding='latin-1', nrows=10000)

#keep only relevant columns: stars, text, useful_dummy. 
df = df_total[['text', 'useful_dummy']]

# Dataframe to use each time we add a new feature. 
df0 = df.copy() 


###--------------------------------------###
###--------CLASSIFIER FUNCTION-----------###
###--------------------------------------###


def run_classifier(data):
    data = data.dropna()
    data = data.drop('text', axis =1)
    y = data['useful_dummy'].values
    X = data.drop('useful_dummy', axis=1).values
    clf = GaussianNB()
    #fit and get score of clf using cross validation
    scoring = ['precision_macro', 'recall_macro', 'balanced_accuracy', 'f1_weighted']
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision_score = scores['test_precision_macro'].mean() 
    recall_score = scores['test_recall_macro'].mean() 
    accuracy_score = scores['test_balanced_accuracy'].mean() 
    f1_score = scores['test_f1_weighted'].mean() 
    return precision_score, recall_score, accuracy_score, f1_score


###--------------------------------------###
###--------------num char----------------###
###--------------------------------------###


#number of words
def word_count(text):
    words = text.split()
    return len(words)


#ADDING FEATURES TO DF
df['num_chars'] = df['text'].apply(len)


#check classifier accuracy
score_num_char = run_classifier(df)


###--------------------------------------###
###-------------POS FEATURES-------------###
###--------------------------------------###

df = df0.copy()

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
df = df[['useful_dummy', 'text', 'VBD']]

#check classifier accuracy
score_VBD = run_classifier(df)


###--------------------------------------###
###-------------REVIEW COUNT-------------###
###--------------------------------------###

df = df0.copy()

### Rating deviance column ###
#Mapping the business rating values into the df 
df['review_count'] = df_total['review_count']
#df=df[['rating_deviance', 'useful_dummy', 'text']]
score_review_count = run_classifier(df)