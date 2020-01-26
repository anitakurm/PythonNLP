"""
doctstring - introduction to script
"""

from collections import Counter
import sys
import getopt
import os
from math import log
import re
import operator
import csv

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.BEST_MODEL = False
    self.stopList = set(self.readFile('data/english.stop'))
    self.numFolds = 10

    self.posText = Counter()   # megatext for pos review, frequency
    self.negText = Counter()   # megatext for neg reviews, frequency
    self.text = {}      # megatext for all reviews
    self.countPosWords   = 0.0    # total number of words in positve megatext
    self.countNegWords   = 0.0    # total number of words in negatvie megatext
    self.countPosReviews = 0.0    # number of positive reviews
    self.countNegReviews = 0.0    # number of negative reviews
    
    self.train_vocab = {}
    self.shared_vocab = {}



    


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets. 
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the 
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingl
  # 
  # Hint: Use filterStopWords(words) defined below


  def addNegationFeatures(self, words):

  ### add your code here 

    pass


  def classify(self, words):

    """
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    n_pos = len(os.listdir("data/imdb1/pos"))
    n_neg = len(os.listdir("data/imdb1/neg"))
    #countTotalReviews = len(os.listdir("data/imdb1/neg") + os.listdir("data/imdb1/pos"))
    probPos = n_pos / (n_pos + n_neg) # Prior of positive reviews
    probNeg = n_neg / (n_pos + n_neg) # Prior of negative reviews
    
    
    pos_words = self.posText.keys()      # Get list of all positive words from self.posText
    neg_words = self.negText.keys()      # Get list of all negative words from self.negText
    self.shared_vocab = set(pos_words) & set(neg_words) # Get intersection of pos and neg words 
    self.train_vocab  = set(pos_words) | set(neg_words) # Get union of pos and neg words from training set 

  


    words = list(set(words)) if self.BOOLEAN_NB else words # Implement BOOLEAN_NB here 
    # words = addNegationFeatures(words) if not self.BEST_MODEL else words # Implement BEST_MODEL here 
    words = self.filterStopWords(words) if self.FILTER_STOP_WORDS else words # Implement stop word filtering here 


    ## HER

    vocab_size = len(set(words)) # Total vocab size 
    pos_factors = # MLE positive (with Laplace smoothing) 
    neg_factors   # MLE negative (with Laplace smoothing) 
    pos_guess = sum(pos_factors)
    neg_guess = sum(neg_factors) 

    

    return 'pos' if pos_guess > neg_guess else 'neg'



  def addExample(self, klass, words):
    """
    words (list): list of strings.
    (Most confusing name ever!! Please fix)
    
    Train model on document with label klass ('pos' or 'neg')
    """
    words = list(set(words)) if self.BOOLEAN_NB else words # Implement BOOLEAN_NB here 
    # words = addNegationFeatures(words) if not self.BEST_MODEL else words # Implement BEST_MODEL here 
    words = self.filterStopWords(words) if self.FILTER_STOP_WORDS else words # Implement stop word filtering here 

    # update vocab
    if klass == 'pos':
      self.posText += Counter(words)
    elif klass == 'neg':
      self.negText += Counter(words)
    
    n_pos = len(os.listdir("data/imdb1/pos"))
    n_neg = len(os.listdir("data/imdb1/neg"))
    n_doc = n_pos + n_neg

    logprior_pos = log(n_pos / (n_pos + n_neg)) # Prior of positive reviews
    logprior_pos = log(n_neg / (n_pos + n_neg)) # Prior of negative reviews
    
    pos_words = self.posText.keys()      # Get list of all positive words from self.posText
    neg_words = self.negText.keys()      # Get list of all negative words from self.negText
    vocab  = set(pos_words) | set(neg_words) # Get union of pos and neg words from training set 

    for word in vocab:
      loglike_pos = log((self.posText[word] + 1)/(sum(self.posText.values()) + len(vocab))) #loglikehood with Laplace smoothing

    for word in vocab:
      loglike_neg = log((self.posText[word] + 1)/(sum(self.posText.values()) + len(vocab))) #loglikehood with Laplace smoothing

    # self.NaiveBayesPerf = something
    ### Your model training code here

  def filterStopWords(self, words):
    """
    Filters stop words
    """
    return [w for w in words if not in self.stopList]
    
      
  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words = self.filterStopWords(words)
      self.addExample(example.klass, words)
      if self.BEST_MODEL:
        words = self.addNegationFeatures(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      yield split

  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      guess = self.classify(words)
      labels.append(guess)
    return labels
  
  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    accuracy = 0.0
    # this is where we should filter all the words, before we pass them in to the trainer and classifier
    # and passing them in one at a time is stupid
    for example in split.train:
      words = example.words 
      classifier.addExample(example.klass, words)
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) )
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  classifier.BEST_MODEL = BEST_MODEL
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print(classifier.classify(testFile))
    
def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  BEST_MODEL = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  elif ('-m','') in options:
    BEST_MODEL = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)

if __name__ == "__main__":
    main()




