# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:28:42 2019

@author: maria
"""

import os 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#os.chdir("/Users/maria/OneDrive/Dokumenter/Yelp") # WINDOWS
os.chdir("/Users/mariaa.madsen/Documents/Yelp") # MAC


# Importing the business data 

import json
business = []
for line in open('business.json', 'r', encoding="utf8"):
    business.append(json.loads(line))

# converting from json to pandas dataframe
from pandas.io.json import json_normalize
business_df = pd.DataFrame.from_dict(json_normalize(business), orient='columns')

# Importing the Yelp subset data 
df = pd.read_excel("/Users/mariaa.madsen/Google Drive/NLP Anita and Maria/Data/yelp_subset.xlsx", sep=";", encoding='latin')
df.dropna()
print(df.head())



###--------------------------------------###
###-------------REVIEW COUNT ------------###
###--------------------------------------###


#Mapping the business rating values into the df 
reviews = zip(business_df["business_id"], business_df["review_count"])
reviewcount_dict = dict(reviews)
df['review_count'] = df['business_id'].map(reviewcount_dict)



###--------------------------------------###
###-----------RATING DEVIANCE------------###
###--------------------------------------###

# Creating a dictionary with each business ID and its star rating 
business_rating = zip(business_df["business_id"], business_df["stars"])
business_dict = dict(business_rating) 

#Mapping the business rating values into the df 
df['business_rating'] = df['business_id'].map(business_dict)

df['rating_deviance'] = df['stars']-df['business_rating']

# Writing the subset dataset as a csv 
#df.to_excel('yelp_subset.xlsx')