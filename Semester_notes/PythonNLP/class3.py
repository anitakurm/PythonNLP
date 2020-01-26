#import

import nltk, nltk.lm, nltk.corpus
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.util import ngrams, everygrams


#Change this later to load the training file
nltk.corpus..sents('.txt')