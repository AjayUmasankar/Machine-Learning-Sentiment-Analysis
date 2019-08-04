'''
Created on 26 Jul 2019

@author: Ajay
'''
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sentence_array = []
topic_array = []
sentiment_array = []
f = open('dataset.tsv', "r", encoding='UTF-8')
for line in f.readlines():
    #line = line[2:]
    line.rstrip()
    words = line.split('\t')
    sentence_array.append(words[1])
    topic_array.append(words[2])
    sentiment_array.append(words[3])
    



for i, sentence in enumerate(sentence_array):
    sentence = re.sub("http.*?(\s|$)", " ",sentence)
    sentence = re.sub("[.\\\,<>\[\]\|\/!\^&‘’\*;:{}=\-\'`~()\"”“,+\?—…–]", "", sentence) # $, #, @, _, %
    sentence_array[i] = sentence
    #print (sentence)
text_data = np.array(sentence_array)


# for i in range(divider,len(y)):
#     sentence = [sentence_array[i]]
#     test = count.transform(sentence).toarray()
#     print(model.predict(test))


#
class preprocessor (object):
    def __init__(self, divider, which, model):
        if(which == "topic"):
            self.y = np.array(topic_array)
        else:
            self.y = np.array(sentiment_array)
        if(model == "dt"):
            self.count = CountVectorizer(token_pattern='[^\s]+', lowercase=False, max_features=200)
        else:
            self.count = CountVectorizer(token_pattern='[^\s]+', lowercase=False)
        self.bag_of_words = self.count.fit_transform(text_data)
        print(self.count.get_feature_names())
        self.X = self.bag_of_words.toarray()
        self.divider = divider
        self.X_train = self.X[:divider]
        self.y_train = self.y[:divider]
        self.X_test = self.X[divider:]
        self.y_test = self.y[divider:]
        self.sentence_array = sentence_array
        
#     def X_train(self):
#         return self.X_train
#     
#     def y_train(self):
#         return self.y_train
#     
#     def X_test(self):
#         return self.X_test
#     
#     def y_test(self):
#         return self.y_test
    

