# requests and xmltodict packages are not native, you migh want to install them
# in your terminal "pip install requests xmltodict"
# if using anaconda "conda install requests xmltodict"

import requests, xmltodict, pickle, os, random
import re
from nltk.corpus import cmudict
arpabet = cmudict.dict()
from nltk.metrics import edit_distance

def category(category):
    """""
        YOU DON'T NEED TO CHANGE THIS FUNCTION, ONLY USE IT!!!
        This function mines the thesaurus rex for a list of words related to a chosen category.
        Note that it saves a cache of the list to avoid downloading all the time.
    """""
    file_path = "./"+category+".pkl"
    if os.path.isfile(file_path): # load stored results
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    url = "http://ngrams.ucd.ie/therex3/common-nouns/head.action?head="+category+"&xml=true"
    response = requests.get(url)
    result = xmltodict.parse(response.content)
    _root_content = result['HeadData']
    print(_root_content)
    result_dict = dict(map(lambda r: tuple([r['#text'].replace('_', '').strip(), int(r['@weight'])]),
                         _root_content['Members']['Member']))
    result_list = list(result_dict.keys())

    with open(file_path, 'wb') as f:  # store the results locally (as a cache)
        pickle.dump(result_list, f, pickle.HIGHEST_PROTOCOL)

    return result_list


def pronounce(word):
    """""
        YOU DON'T NEED TO CHANGE THIS FUNCTION, ONLY USE IT!!!
        This function looks into the cmu dictionnary to translate a word into its phonetic spelling.
        Input a string, outputs a list of phonemes
    """""
    return arpabet[word.lower()][0] if word.lower() in arpabet else None # make sure the word is lowercased and exists in the dictionary


def pun(sent,cat):
    """""
        THIS IS THE FUNCTION YOU HAVE TO WRITE
        It takes an expression and a category as input,
        Chooses a word in the expression,
        Find a word related to the category that sounds similar,
        replace the word chosen by this similar sounding word,
        to build a new expression
    """""
    #first slice the string and choose a word in it, maybe you need to clean it from punctuation?
    sent = re.sub(r'[^\w\s]', '', sent) #everything that is not a word - replace with nothing
    sent = sent.split() #split into words
    word_of_interest = sent[2]
 


    #second, load the category as a list- drink or food
    cat_list = category(cat)
    print(cat_list)


    #careful, not all words are in the spelling dictionnary, make sure to only keep those that are        
    if word_of_interest in arpabet:
        word_of_interest = word_of_interest
    else:
            print("word not found in dictionary")
            return False

    #third, translate the list of words into a list of their phonetic representation
    translated_category = [pronounce(word) for word in cat_list]

    #fourth, create a list of distances (use the edit_distance function from nltk)
    distance_list = [edit_distance(word, translated_category) for word in translated_category]
    print(distance_list)
   


    #fifth, create a list of words with the minimum distance from the target word

    #sixth, choose a substitute in the list created above


    #seventh, replace the target word with its substitute


    #rebuild a string out of the list


teststr = "This is the test, sentence!"
re.sub(r'[^\w\s]', '', teststr) #everything that is not a word - replace with nothing
teststr = teststr.split()
print(teststr)

sentence = input("enter any kind of idioma:")
print(pun(sentence,"drink"))


pun("this is a big tigers! and lion!", "cat")