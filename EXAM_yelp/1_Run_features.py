#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:38:42 2019

@author: mariaa.madsen
"""
#%%
###--------------------------------------###
###---------------SET UP-----------------###
###--------------------------------------###

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
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
df_total = pd.read_excel('yelp_subset.xlsx', sep=';', encoding='latin-1', nrows=100)      #subset where not useful<2 and useful >1
#df_total = pd.read_excel('yelp_subset2.xlsx', sep=';', encoding='latin-1', nrows=10000)    #subset where not useful=0 and useful >2

#keep only relevant columns: stars, text, useful_dummy. 
df = df_total[['stars','text', 'useful_dummy']]

#%%

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

#%%

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

#%%

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

#%%

###--------------------------------------###
###-------------REVIEW COUNT-------------###
###--------------------------------------###

### Rating deviance column ###
#Mapping the business rating values into the df 
df['review_count'] = df_total['review_count']

#%%

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

#%%

###--------------------------------------###
###----------SENTIMENT ANALYIS-----------###
###--------------TextBlob----------------###

def detect_polarity(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df.text.apply(detect_polarity)

def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

df['subjectivity'] = df.text.apply(detect_subjectivity)

#%%

###--------------------------------------###
###------------RATING DEVIANCE-----------###
###--------------------------------------###

### Rating deviance column ###
#Mapping the business rating values into the df 
df['rating_deviance'] = df_total['rating_deviance']

#%%

###--------------------------------------###
###------------Concreteness-------------###
###--------------------------------------###

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

#load concreteness data
conc_database = pd.read_excel('Concreteness_ratings.xlsx', encoding='latin-1')

def get_concreteness(text):
    word_list = text.split(' ')
    word_list = list(lemmatizer.lemmatize(word) for word in word_list)
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    concreteness_score = df_with_concr['Conc.M'].mean()
    return concreteness_score

df['concreteness'] = df['text']. apply(get_concreteness)

#%%

###--------------------------------------###
###-------------Familiarity--------------###
###--------------------------------------###


def get_familiarity(text):
    word_list = text.split(' ')
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    familiarity_score = df_with_concr['Percent_known'].mean()
    return familiarity_score

df['familiarity'] = df['text']. apply(get_familiarity)


#%%

###--------------------------------------###
###--------------Profanity---------------###
###--------------------------------------###

import re
#load concreteness data
tab_database = pd.read_csv('TabooWords.csv', encoding='latin-1')


###---------------Taboo---------------###



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


###---------------Arousal---------------###


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


#Write file with features 
df.to_excel('yelp_subset_features.xlsx')
#df.to_excel('yelp_subset_features_2.xlsx')
#%%%
