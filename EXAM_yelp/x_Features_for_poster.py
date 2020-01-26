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
df_total = pd.read_excel('yelp_subset.xlsx', sep=';', encoding='latin-1', nrows=5000)

#keep only relevant columns: stars, text, useful_dummy. 
df = df_total[['text', 'useful_dummy']]

# Dataframe to use each time we add a new feature. 
df0 = df.copy() 


###--------------------------------------###
###--------CLASSIFIER FUNCTION-----------###
###--------------------------------------###


#def run_classifier(data):
#    data = data.dropna()
#    data = data.drop('text', axis =1)
#    y = data['useful_dummy'].values
#    X = data.drop('useful_dummy', axis=1).values
#    clf = LogisticRegression()
#    #fit and get score of clf using cross validation
#    scoring = ['balanced_accuracy', 'precision_weighted', 'recall_weighted']
#    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
#    precision_score = scores['test_precision_weighted'].mean()
#    recall_score = scores['test_recall_weighted'].mean()
#    accuracy_score = scores['test_balanced_accuracy'].mean() 
#    return accuracy_score, precision_score, recall_score

def run_classifier(data):
    data = data.dropna()
    data = data.drop('text', axis =1)
    y = data['useful_dummy'].values
    X = data.drop('useful_dummy', axis=1).values
    clf = LogisticRegression()
    #fit and get score of clf using cross validation
    scoring = ['accuracy', 'f1', 'balanced_accuracy', 'f1_weighted']
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    accuracy = scores['test_accuracy'].mean() 
    f1 = scores['test_f1'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    return accuracy, f1, accuracy_balanced, f1_weighted

# Adding stars to the df
df['stars']=df_total['stars']

#test classifier on no features except stars and text
score_stars = run_classifier(df)


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


#Additional features to  consider: number of sentences; average length of sentences; presence of all-capital words

#check classifier accuracy
score_basicfeatures = run_classifier(df)


###--------------------------------------###
###----------READABILITY FEATURES--------###
###--------------------------------------###

df = df0.copy()

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

#check classifier accuracy
score_readable = run_classifier(df)


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
df = df.drop(columns=['pos'])

#check classifier accuracy
score_pos = run_classifier(df)


###--------------------------------------###
###-------------REVIEW COUNT-------------###
###--------------------------------------###

df = df0.copy()

### Rating deviance column ###
#Mapping the business rating values into the df 
df['review_count'] = df_total['review_count']
#df=df[['rating_deviance', 'useful_dummy', 'text']]
score_review_count = run_classifier(df)


###--------------------------------------###
###-----------SENTIMENT FEATURES---------###
###-----------------Vader----------------###

df = df0.copy()

#calling the sentiment analysis tool something shorter
analyzer = SentimentIntensityAnalyzer()


def get_vadersentiment(text):
    compound = analyzer.polarity_scores(text)['compound']
    pos = analyzer.polarity_scores(text)['pos']
    neu = analyzer.polarity_scores(text)['neu']
    neg = analyzer.polarity_scores(text)['neg']
    
    return compound, pos, neu, neg


df['sent_compound'], df['sent_pos'], df['sent_neu'], df['sent_neg'] = zip(*df['text']. map(get_vadersentiment))

#check classifier accuracy
score_sentiment_vader = run_classifier(df)



###--------------------------------------###
###----------SENTIMENT ANALYIS-----------###
###--------------TextBlob----------------###

df = df0.copy()

def detect_polarity(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df.text.apply(detect_polarity)

def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

df['subjectivity'] = df.text.apply(detect_subjectivity)

score_sentiment_textblob =run_classifier(df)


###--------------------------------------###
###------------RATING DEVIANCE-----------###
###--------------------------------------###

df = df0.copy()

### Rating deviance column ###
#Mapping the business rating values into the df 
df['rating_deviance'] = df_total['rating_deviance']
#df=df[['rating_deviance', 'useful_dummy', 'text']]
score_rating_deviance = run_classifier(df)


###--------------------------------------###
###------------Concreteness-------------###
###--------------------------------------###

df = df0.copy()

#load concreteness data
conc_database = pd.read_excel('Concreteness_ratings.xlsx', encoding='latin-1')

def get_concreteness(text):
    word_list = text.split(' ')
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    concreteness_score = df_with_concr['Conc.M'].mean()
    return concreteness_score

df['concreteness'] = df['text']. apply(get_concreteness)

score_concreteness = run_classifier(df)


###--------------------------------------###
###-------------Familiarity--------------###
###--------------------------------------###

df = df0.copy()

def get_familiarity(text):
    word_list = text.split(' ')
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    familiarity_score = df_with_concr['Percent_known'].mean()
    return familiarity_score

df['familiarity'] = df['text']. apply(get_familiarity)

score_familiarity = run_classifier(df)



###--------------------------------------###
###--------------Profanity---------------###
###--------------------------------------###

import re
#load concreteness data
tab_database = pd.read_csv('TabooWords.csv', encoding='latin-1')


###---------------Taboo---------------###

df = df0.copy()


def get_taboo(text):
    text = re.sub(r'[^\w\s]','', text)
    word_list = text.split(' ')
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_tab = pd.merge(word_df, tab_database, on="Word")
    taboo_score = df_with_tab['Taboo'].mean()
    if taboo_score > 0:
        return taboo_score
    else:
        return 0

df['taboo'] = df['text']. apply(get_taboo)

score_taboo = run_classifier(df)


###---------------Arousal---------------###

df = df0.copy()

def get_taboo_arousal(text):
    text = re.sub(r'[^\w\s]','', text)
    word_list = text.split(' ')
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_tab = pd.merge(word_df, tab_database, on="Word")
    taboo_arousal = df_with_tab['Arousal'].mean()
    if taboo_arousal > 0:
        return taboo_arousal
    else:
        return 0


df['taboo_arousal'] = df['text']. apply(get_taboo_arousal)

score_taboo_arousal = run_classifier(df)


## Writing a csv file with all the relevant features
relevant_features_list = [score_basicfeatures,score_pos, score_rating_deviance,score_readable, score_review_count, score_sentiment_textblob, score_sentiment_vader, score_stars, score_concreteness, score_familiarity, score_taboo, score_taboo_arousal]
relevant_features_names = ["score_basicfeatures", "score_pos", "score_rating_deviance","score_readable", "score_review_count", "score_sentiment_textblob", "score_sentiment_vader", "score_stars", "score_concreteness", "score_familiarity", "score_taboo", "score_taboo_arousal"]
relevant_features = pd.DataFrame(relevant_features_list, columns =['accuracy', 'f1_score', 'accuracy_balanced', 'f1_weighted'], index=relevant_features_names) 
relevant_features.to_csv('relevant_features.csv')

