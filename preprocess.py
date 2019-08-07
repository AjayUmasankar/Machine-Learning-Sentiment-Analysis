'''
Created on 26 Jul 2019

@author: Ajay
'''
import re
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import defaultdict
import plotly.graph_objects as go

sentence_array = []
topic_array = []
sentiment_array = []
f = open('dataset.tsv', "r", encoding='UTF-8')
numlines = 0
for line in f.readlines():
    #line = line[2:]
    numlines = numlines + 1
    line.rstrip()
    words = line.split('\t')
    sentence_array.append(words[1])
    topic_array.append(words[2])
    sentiment_array.append(words[3])

# Filters Sentence:
    # Treats URL's as space
    # Deletes all non-alphabetic, non-[#@_$% ], non-numeric characters. 
for i, sentence in enumerate(sentence_array):
    sentence = re.sub("http.*?(\s|$)", " ",sentence)
    sentence = re.sub("[^A-Za-z0-9#@_$% ]", "", sentence)
    sentence_array[i] = sentence
    #print (sentence)



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


# CountVectorizer:
# https://piazza.com/class/jvhnwcx8t2o5qg?cid=271 - can use CountVectorizer over FreqDist
    # Token pattern: Words = 2+ non-space characters 
    # lowercase: CAPITAL words differ from lowercase
    # max_features: n most frequent words from the vocabulary
class preprocessor (object):
    def __init__(self, divider, which, model):
        # divider = number of elements in training set
        divider = 1500
        ps = PorterStemmer() 
        my_stop_words = []
        stop_words = (stopwords.words('english')) 
        for w in stop_words:
            my_stop_words.append(ps.stem(w))

        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([ps.stem(w) for w in analyzer(doc)])
        if(which == "topic"):
            self.y = np.array(topic_array)
        else:
            self.y = np.array(sentiment_array)
        if(model == "dt"):
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=False, max_features=200)
        else:   # can use token_pattern='([@#$%_A-Za-z0-9]{2,})' also 
            self.count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=False)
            self.count = StemmedCountVectorizer(token_pattern='([^\s]{2,})', lowercase=False, analyzer="word", stop_words=stopwords.words('english'))

        # Stemming words in sentence, double stemming doesnt do anything
        for sentence in sentence_array:
            new = ""
            for word in word_tokenize(sentence):
                new = new + " " + ps.stem(word)
            #print(new)
             
        # Creating bag of words
        text_data = np.array(sentence_array)
        start = time.time()
        self.bag_of_words = self.count.fit_transform(text_data)
        self.X = self.bag_of_words.toarray()
        stop = time.time()
        print('Time: ', stop - start)  
        #print("Divider is", divider)
        self.X_train = self.X[:divider]
        self.y_train = self.y[:divider]
        self.X_test = self.X[divider:]
        self.y_test = self.y[divider:]
        self.sentence_array = sentence_array
        print(self.count.get_feature_names())
        
        # VADER Analysis
        analyser = SentimentIntensityAnalyzer()
        vader_results = []
        for sentence in self.sentence_array[divider:]:
            score = analyser.polarity_scores(sentence)
            compound = score.get('compound')
            if (compound >= 0.05):
                vader_results.append("positive")
            elif (compound <= -0.05):
                vader_results.append("negative")
            else:
                vader_results.append("neutral")
        print(classification_report(self.y_test, np.array(vader_results)))
        print("Accuracy:  ", accuracy_score(self.y_test, np.array(vader_results)))
        print("Precision: ", precision_score(self.y_test, np.array(vader_results), average='macro'))
        print("Recall:    ", recall_score(self.y_test, np.array(vader_results), average='macro'))
        print("###################### Preprocessing end #########################")

#         print(self.bag_of_words)
#         visualizer = FreqDistVisualizer(features=self.count.get_feature_names())
#         visualizer.fit(self.bag_of_words)
#         visualizer.poof()

    



