#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code appendix for exam project in the NLP course on the Cognitive Science master's programme  

Authors: Anita Kurm and Maria Abildtrup Madsen
"""

#%%
###--------------------------------------###
###---------------SET UP-----------------###
###--------------------------------------###

import os 
import pandas as pd
import json
from pandas.io.json import json_normalize
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Remove warnings 
pd.set_option('mode.chained_assignment', None) # Remove warnings 


# SKLearn modules 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

# NLTK modules 
import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

# Modules for visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Other modules for feature selection
from textatistic import Textatistic #for readability metric
from collections import Counter
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Setting the working directory
os.chdir("/Users/maria/OneDrive/Dokumenter/Yelp") # Maria 
#os.chdir("/Users/anita/OneDrive/Dokumenter/Yelp") # Anita

#%%

###--------------------------------------###
###----------DATA PREPERATION------------###
###--------------------------------------###

# Importing the json file with review data
reviews = []
for line in open('review.json', 'r', encoding="utf8"):
    reviews.append(json.loads(line))

# converting from json to pandas dataframe
    from pandas.io.json import json_normalize
df = pd.DataFrame.from_dict(json_normalize(reviews), orient='columns')

# Creating a dummy variable where overall rating from 3-5: 1, else: 0 
df = df.assign(useful_dummy = df['useful'])
df.loc[df['useful']>1, 'useful_dummy'] = 1
df.loc[df['useful']<2, 'useful_dummy'] = 0
print(df.head())

#----Adding a column in the df with rating deviance, calculated from the business dataset---#

# Importing the business data 
business = []
for line in open('/Users/mariaa.madsen/Documents/Yelp/business.json', 'r', encoding="utf8"):
    business.append(json.loads(line))

# converting from json to pandas dataframe
business_df = pd.DataFrame.from_dict(json_normalize(business), orient='columns')

# Creating a dictionary with each business ID and its star rating 
business_rating = zip(business_df["business_id"], business_df["stars"])
business_dict = dict(business_rating) 

#Mapping the business rating values into the df 
df['business_rating'] = df['business_id'].map(business_dict)
df['rating_deviance'] = df['stars']-df['business_rating']


#----Adding a column in the df with review count, calculated from the business dataset----#

#Mapping the business rating values into the df 
reviews = zip(business_df["business_id"], business_df["review_count"])
reviewcount_dict = dict(reviews)
df['review_count'] = df['business_id'].map(reviewcount_dict)

#%%

#----CREATING A SUBSET---#

# Making a random sample of the dataset 
df2 = df.sample(frac=0.01, replace=True, random_state=1)
# Writing the subset dataset as a csv 
df2.to_excel(r'yelp_subset.xlsx', sep=";")

# Setting working directory 
os.chdir("/Users/mariaa.madsen/Google Drive/NLP Anita and Maria/Data")

#read data. 
df_total = pd.read_excel('yelp_subset.xlsx', sep=';', encoding='latin-1', nrows=10000)      #subset where not useful<2 and useful >1

#keep only relevant columns: stars, text, useful_dummy. 
df = df_total[['stars','text', 'useful_dummy']]

#%%
""" FEATURE ENGINEERING:"""

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

#load concreteness data
conc_database = pd.read_excel('Concreteness_ratings.xlsx', encoding='latin-1')

def get_concreteness(text):
    word_list = text.split(' ')
    word_list = lemmatizer.lemmatize(word_list)
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
    word_list = lemmatizer.lemmatize(word_list)
    word_df = pd.DataFrame(word_list, columns = ['Word'])
    df_with_concr = pd.merge(word_df, conc_database, on="Word")
    familiarity_score = df_with_concr['Percent_known'].mean()
    return familiarity_score

df['familiarity'] = df['text']. apply(get_familiarity)


#%%

###--------------------------------------###
###--------------Profanity---------------###
###--------------------------------------###

#load concreteness data
tab_database = pd.read_csv('TabooWords.csv', encoding='latin-1')


###---------------Taboo---------------###



def get_taboo(text):
    text = re.sub(r'[^\w\s]','', text)
    word_list = text.split(' ')
    word_list = lemmatizer.lemmatize(word_list)
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
    word_list = lemmatizer.lemmatize(word_list)
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

#%% 

""" FEATURE SELECTION, CLASSIFICATION AND MODEL EVALUATION """ 

###--------------------------------------###
###------------DATA PREP-----------------###
###--------------------------------------###

#Read data and keep only 10000 rows for computational reasons
df = pd.read_excel('yelp_subset_features.xlsx', sep=';', encoding='latin-1', nrows=10000)

# Preparing data
data = df.dropna()
data = data.drop('text', axis =1)

y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values


# Calulating correlations 
cor = data.corr()
cor_target = cor["useful_dummy"]
print(cor_target)
matrix = np.triu(data.corr())
ax = sns.heatmap(data.corr(), square=True, mask = matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top - 1)
side, endside = ax.get_xlim()
ax.set_xlim(side- 1, endside +1)
plt.yticks(rotation='horizontal')
plt.title('All features')


###--------------------------------------###
###------------CROSS VALIDATION ---------###
###--------------------------------------###
# create classifiers
names = ["3-Nearest Neighbors", 
        "5-Nearest Neighbors",
        "7-Nearest Neighbors",
        "Logistic Regression",
        "Logistic Regression weighted classes",
        "SVM (gamma=auto)",
        "Decision Tree (max_depth = None)", 
        "Random Forest (max_depth = None, trees = 100)", 
        "Random Forest (max_depth = 2)",
        "Naive Bayes (priors = [0.75, 0.25])",
        "Naive Bayes no priors",
        "LDA"]
         
classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(7),
    LogisticRegression(),
    LogisticRegression(class_weight='balanced'),
    SVC(gamma='auto'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    RandomForestClassifier(max_depth=2, random_state=0),
    GaussianNB(priors = [0.75, 0.25]),
    GaussianNB(),
    LDA()
    ]

# defines scoring measures
scoring = {'precision_macro': 'precision_macro',
           'recall_macro': 'recall_macro', 
           'balanced_accuracy': 'balanced_accuracy', 
           'f1_weighted': 'f1_weighted'
           }


results_all = []
for clf, clfname in zip(classifiers, names):
    print(f"Cross-validating classifier: \t", clfname)
    start = time.time()
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision = scores['test_precision_macro'].mean() 
    recall = scores['test_recall_macro'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    results= clfname, precision, recall, accuracy_balanced, f1_weighted
    results_all.append(results)
    end = time.time() - start
    print(" - time taken: ", round(end, 3))

results_all
all_results = pd.DataFrame(results_all, columns = ['Classifier','Precision', 'Recall', 'Accuracy balanced', 'F1 weighted'])
all_results.to_csv('classifier_performances.csv')


###--------------------------------------###
###------------FEATURE SELECTION---------###
###--------------------------------------###


#----------- method = LASSO CV -------------#

y = data['useful_dummy'].values
X = data.drop('useful_dummy', axis=1).values
from sklearn.linear_model import LassoCV
clf = LassoCV()
# Set a minimum threshold of 0.25
clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
list_true = model.get_support()

# Returns a dataframe with chosen features     
df_features = data.drop('useful_dummy', axis=1)
feat_bool = pd.DataFrame(list_true, df_features.columns, columns=['chosen'])
chosen_features = feat_bool[feat_bool['chosen']==True] 
needed_data = data[[i for i in chosen_features.index]]
needed_data['useful_dummy'] = data['useful_dummy']

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# Calulating correlations 
cor = needed_data.corr()
cor_target = cor["useful_dummy"]
print(cor_target)
matrix = np.triu(needed_data.corr())
ax = sns.heatmap(needed_data.corr(), annot = True, square=True, mask = matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 1, top - 1)
side, endside = ax.get_xlim()
ax.set_xlim(side- 1, endside +1)
plt.yticks(rotation='horizontal')
plt.title('Features selected via Lasso')


### RUNNING CROSS validation on Lasso features (needed data)
y = needed_data['useful_dummy'].values
X = needed_data.drop('useful_dummy', axis=1).values

results_lasso_features = []
for clf, clfname in zip(classifiers, names):
    print(f"Cross-validating classifier: \t", clfname)
    start = time.time()
    scores = cross_validate(clf, X, y, scoring=scoring, cv = 5)
    precision = scores['test_precision_macro'].mean() 
    recall = scores['test_recall_macro'].mean() 
    accuracy_balanced = scores['test_balanced_accuracy'].mean() 
    f1_weighted = scores['test_f1_weighted'].mean() 
    results= clfname, precision, recall, accuracy_balanced, f1_weighted
    results_lasso_features.append(results)
    end = time.time() - start
    print(" - time taken: ", round(end, 3))

results_lasso_features
lasso_results = pd.DataFrame(results_lasso_features, columns = ['Classifier','Precision', 'Recall', 'Accuracy balanced', 'F1 weighted'])
lasso_results.to_csv('lasso_features_performances.csv')
