'''
Created on 26 Jul 2019

@author: Ajay
'''
import re
import numpy as np
import time
import sys
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import defaultdict
import plotly.graph_objects as go
from nltk.stem.snowball import EnglishStemmer

divider_outer = 1500
instance_array = []
sentence_array = []
topic_array = []
sentiment_array = []
f = open(sys.argv[1], "r", encoding='UTF-8')
# print(sys.argv[1])
numlines = 0
for line in f.readlines():
    line.rstrip()
    words = line.split('\t')
    instance_array.append(words[0])
    sentence_array.append(words[1])
    topic_array.append(words[2])
    sentiment_array.append(words[3])
    numlines = numlines + 1
# print("Sentence Array Size: ", len(sentence_array))

divider_outer=numlines
numlines = 0
f = open(sys.argv[2], "r", encoding='UTF-8')
for line in f.readlines():
    line.rstrip()
    words = line.split('\t')
    instance_array.append(words[0])
    sentence_array.append(words[1])
    numlines = numlines + 1
# print("Test set lines: ", numlines)


# Filters Sentence:
    # Treats URL's as space
    # Deletes all non-alphabetic, non-[#@_$% ], non-numeric characters. 
for i, sentence in enumerate(sentence_array):
    sentence = re.sub("https?://.*?(\s|$)", " ",sentence)
    sentence = re.sub("[^A-Za-z0-9#@_$% ]", "", sentence)
    sentence_array[i] = sentence




# CountVectorizer:
# https://piazza.com/class/jvhnwcx8t2o5qg?cid=271 - can use CountVectorizer over FreqDist
    # Token pattern: Words = 2+ non-space characters 
    # lowercase: CAPITAL words differ from lowercase
    # max_features: n most frequent words from the vocabulary
class preprocessor (object):
    def __init__(self, which, model):
        # divider = number of elements in training set
        divider=divider_outer
#         print("divider is: ", divider)
        if(which == "topic"):
            self.y = np.array(topic_array)
        else:
            self.y = np.array(sentiment_array)
        if(model == "dt"):
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=False, max_features=200)
        elif(model == "mysentiment"):   # can use token_pattern='([@#$%_A-Za-z0-9]{2,})' also 
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=True)
                # Stemming 
                #         ps = EnglishStemmer() 
                #         for i, sentence in enumerate(sentence_array):
                # #             print(sentence)
                #             new = ""
                #             for word in sentence.split(" "):
                #                 stemmed = ps.stem(word)
                #                 newword = word[:len(stemmed)]
                #                 new = new + " " + newword
                #             sentence_array[i] = new.lstrip()
                # #             print(new.lstrip())
    
        elif(model == "mytopic"):
            self.count = CountVectorizer(token_pattern='([^\s]{2,})')
        else:
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=False) 
            

#         # Stemming + Stopword removal
#         ps = PorterStemmer() 
#         for i, sentence in enumerate(sentence_array):
#             print(sentence)
#             new = ""
#             for word in sentence.split(" "):
#                 lower_word = word.lower()
#                 if(lower_word not in set(stopwords.words('english'))):
#                     stemmed = ps.stem(word)
#                     newword = word[:len(stemmed)]
#                     new = new + " " + newword
#             sentence_array[i] = new.lstrip()
#             print(new.lstrip())

           
        # Creating bag of words  
        # Creating X_ and Y_ train and test from bag of words
        text_data = np.array(sentence_array)
        self.bag_of_words = self.count.fit_transform(text_data)
        self.X = self.bag_of_words.toarray()
        self.X_train = self.X[:divider]
        self.y_train = self.y[:divider]
        self.X_test = self.X[divider:]
        self.y_test = self.y[divider:]
        self.sentence_array = sentence_array
        self.instance_array = instance_array
        self.divider = divider_outer
#         print("###################### Preprocessing end #########################")
