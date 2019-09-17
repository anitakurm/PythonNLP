
#import all data from the book
from nltk.book import *

#check what text it is
text6

#searching text for match
#concordance - show words in context
text1.concordance("monstrous")

#searching another text for another word
text2.concordance("affection")
text5.concordance("lol")


#common_contexts - examine just the contexts shared by the specified words
text2.common_contexts(["monstrous", "very"])

#find words with similar context
text5.similar("love")
text5.common_contexts(["love","hope"])


#count occurrences
text5.count("lol")
#in %
100 * text5.count("lol")/len(text5)


#making your own function:

def percentage(count,total):
    return 100 * count / total

percentage(4, 5)
percentage(text4.count('a'), len(text4))


#defining a list of words
ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
len(ex1)

ex2 = ['Some', 'body', 'to', 'love']

ex3 = ex1+ex2

ex3
len(ex3)

ex3.append("likey")
ex3


#Indexing
ex3[0]
ex3[0:4]
ex3[:4] #same same

#modifying using index
ex3[:2] = ['Monty2','Python2']
ex3

ex3[4:8] #11 wordss
ex3[4:8] = ['Holygrail',"Somebody"]


#Strings - a list of characters
name = 'Monty'

name[0]

#multiplication and addition with strings
name * 2

name + '!'

#join the words of the list into a single string
' '.join(['Monty','Python'])

#split the string into a list
'Monty Python'.split()


####Chapter 3
saying = ['After', 'all', 'is', 'said', 'and', 'done', 'more', 'is', 'said', 'than', 'done']

tokens = set(saying)
tokens = sorted(tokens)
tokens[-2:]


#frequency distribution
fdist1 = FreqDist(text1)
print(fdist1)
fdist1.most_common(50)

fdist1['whale']

fdist2 = FreqDist(text2)
fdist2.most_common(10)

fdist2['in']

#Fine-grained selection of Words
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

sorted(w for w in set(text5) if len(w) > 7 and FreqDist(text5)[w]>7)



#Collocations and bigrams
list(bigrams(['more','is', 'said', 'than', 'done']))

#collocations are frequent bigrams
text8.collocation_list()


#counting other things
[len(w) for w in text1]

fdist = FreqDist(len(w) for w in text1)

#the result is a distribution
#containing a quarter a million items
#each is a number corresponding to a word token in the text
print(fdist)

fdist



#word compariosn operators
sorted(w for w in set(text1) if w.endswith('ableness'))
sorted(term for term in set(text4) if 'gnt' in term)
sorted(item for item in set(text6) if item.istitle())
sorted(item for item in set(sent7) if item.isdigit())

sorted(w for w in set(text7) if '-' in w and 'index' in w)
sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10)
sorted(w for w in set(sent7) if not w.islower())
sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)


###Operating on every element

#e.g. finding length of every word
[len(w) for w in text1]

#convert every word to uppercase
[w.upper() for w in text1]




#IF statements
word = 'cat'
if len(word) < 5:
    print('word length is less than 5')

for word in ['Call', 'me', 'Ishmael','.']:
    print(word)


#Looping with conditions
sent1 = ['Call', 'me', 'Ishmael','.']
for xyyy in sent1:
    if xyyy.endswith('l'):
        print(xyyy)


for token in sent1:
    if token.islower():
        print(token, 'is a lowercase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')


#we can also ask Python to return words in the same line
#instead of new row per item in the loop, using end = ' '
tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')