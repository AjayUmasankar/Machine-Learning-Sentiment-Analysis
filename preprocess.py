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
major_sentiment = []
major_topic = []
instance_array = []
sentence_array = []
topic_array = []
sentiment_array = []
f = open(sys.argv[1], "r", encoding='UTF-8')
# print(sys.argv[1])
numlines = 0
for line in f.readlines():
    #line = line[2:]
    line.rstrip()
    words = line.split('\t')
#     if(words[3] != "neutral"):
    instance_array.append(words[0])
    sentence_array.append(words[1])
    topic_array.append(words[2])
    sentiment_array.append(words[3])
    major_sentiment.append("negative")
    major_topic.append("10003")
#     if(numlines == 1499): # 1500 lines processed
#         divider_outer=len(sentence_array)
#         print(divider_outer)
    numlines = numlines + 1

print("Sentence Array Size: ", len(sentence_array))

# Filters Sentence:
    # Treats URL's as space
    # Deletes all non-alphabetic, non-[#@_$% ], non-numeric characters. 
for i, sentence in enumerate(sentence_array):
#     sentence = re.sub("http.*?(\s|$)", " ",sentence)
    sentence = re.sub("https?://.*?(\s|$)", " ",sentence)
    sentence = re.sub("[^A-Za-z0-9#@_$% ]", "", sentence)
    sentence_array[i] = sentence
    #print (sentence)
#print(len(sentence_array))


# FreqDistribution of Sentiments/Topics
    # Sentiments
    # sent_dict = defaultdict(int)
    # for sentiment in sentiment_array:
    #     sent_dict[sentiment] = sent_dict[sentiment] + 1
    #     
    # sent_sum = 0
    # x=[]
    # y=[]
    # for sentiment in sent_dict:
    #     sent_sum += sent_dict[sentiment]
    #     x.append(sentiment)
    #     y.append(sent_dict[sentiment])
    #     #print(sentiment, sent_dict[sentiment])
    # print(sent_sum)
    # 
    # fig = go.Figure(
    #     data=[go.Bar(x=x, y=y, text=y, textposition='auto')],
    #     layout=dict(title=dict(text="Frequency Distribution of Sentiments"),
    #                 xaxis=dict(title=dict(text="Sentiment Type"), tickmode='linear'),
    #                 yaxis=dict(title=dict(text="Frequency")))
    # )
    # fig.show()
    # 
    #     # Topics
    # topic_dict = defaultdict(int)
    # for topic in topic_array:
    #     topic = int(topic)-10000
    #     topic_dict[topic] = topic_dict[topic] + 1
    # topic_sum = 0
    # x=[]
    # y=[]
    # for topic in topic_dict:
    #     #topic = int(topic)-10000
    #     x.append(topic)
    #     y.append(topic_dict[topic])
    #     topic_sum += topic_dict[topic]
    # print(topic_sum)
    # 
    # fig = go.Figure(
    #     data=[go.Bar(x=x, y=y, text=y, textposition='auto')],
    #     layout=dict(title=dict(text="Frequency Distribution of Topics"),
    #                 xaxis=dict(title=dict(text="Topic ID (0-19)"), tickmode='linear'),
    #                 yaxis=dict(title=dict(text="Frequency")))
    # )
    # fig.show()



# VADER Analysis
analyser = SentimentIntensityAnalyzer()
vader_results = []
for sentence in sentence_array[divider_outer:]:
    score = analyser.polarity_scores(sentence)
    compound = score.get('compound')
    if (compound >= 0.05):
        vader_results.append("positive")
    elif (compound <= -0.05):
        vader_results.append("negative")
    else:
        vader_results.append("neutral")
#print(vader_results)
vader_array = np.array(vader_results)

# CountVectorizer:
# https://piazza.com/class/jvhnwcx8t2o5qg?cid=271 - can use CountVectorizer over FreqDist
    # Token pattern: Words = 2+ non-space characters 
    # lowercase: CAPITAL words differ from lowercase
    # max_features: n most frequent words from the vocabulary
class preprocessor (object):
    def __init__(self, divider, which, model):
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
#             stopwords=['#auspol', '#ausvot', 'and', 'be', 'in' ,'is', 'it', 'not', 'of', 'on', 'that', 'the', 'tp', 'will', 'for', 'have',' job', 'are', 'about', 'with', 'you', 'say']
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=True, max_features=900, stop_words='english')
            ps = PorterStemmer() 
            for i, sentence in enumerate(sentence_array):
                new = ""
                for word in sentence.split(" "):
                    stemmed = ps.stem(word)
                    newword = word[:len(stemmed)]
                    new = new + " " + newword
                sentence_array[i] = new.lstrip()
        elif(model == "mytopic"):   # can use token_pattern='([@#$%_A-Za-z0-9]{2,})' also 
#             stopwords=['#auspol', '#ausvot', 'and', 'be', 'in' ,'is', 'it', 'not', 'of', 'on', 'that', 'the', 'tp', 'will', 'for', 'have',' job', 'are', 'about', 'with', 'you', 'say']
            self.count = CountVectorizer(token_pattern='([^\s]{2,})')
            ps = EnglishStemmer() 
            for i, sentence in enumerate(sentence_array):
                new = ""
                for word in sentence.split(" "):
                    stemmed = ps.stem(word)
                    newword = word[:len(stemmed)]
                    new = new + " " + newword
                sentence_array[i] = new.lstrip()
        else:
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=False) 
            

#         # Stemming + Stopword removal
#         ps = EnglishStemmer() 
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
#         start = time.time()
        self.bag_of_words = self.count.fit_transform(text_data)
        self.X = self.bag_of_words.toarray()
#         stop = time.time()
#         print('Time: ', stop - start)  
        self.X_train = self.X[:divider]
        self.y_train = self.y[:divider]
        self.X_test = self.X[divider:]
        self.y_test = self.y[divider:]
        print("Test set size: ", len(self.X_test))
        self.sentence_array = sentence_array
        self.instance_array = instance_array
        print(self.count.get_feature_names())
        
#         print("######################## MAJORITY CLASS ##########################")
#         if(which == "sentiment"):
#             majority_array = np.array(major_sentiment[divider:])
#         else:
#             majority_array = np.array(major_topic[divider:])
#         print(classification_report(self.y_test, majority_array))
#         print("Accuracy:  ", accuracy_score(self.y_test, majority_array))
#         print("Precision (array): ", precision_score(self.y_test, majority_array, average=None))
#         print("Precision (macro): ", precision_score(self.y_test, majority_array, average='macro'))
#         print("Recall (macro):    ", recall_score(self.y_test, majority_array, average='macro'))
#     
#         print("########################## VADER ##############################")
#         print(classification_report(self.y_test, vader_array))
#         print("Accuracy:  ", accuracy_score(self.y_test, vader_array))
#         print("Precision (array): ", precision_score(self.y_test, vader_array, average=None))
#         print("Precision (macro): ", precision_score(self.y_test, vader_array, average='macro'))
#         print("Recall (macro):    ", recall_score(self.y_test, vader_array, average='macro'))
        print("###################### Preprocessing end #########################")
