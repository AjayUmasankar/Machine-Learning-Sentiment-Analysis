# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# 
# '''
# Created on 26 Jul 2019
# 
# @author: Ajay
# '''
# 
from preprocess import preprocessor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#GOAL=0.74

sentiment_array = []
sentence_array = []
f = open(sys.argv[1], "r", encoding='UTF-8')
# print(sys.argv[1])
numlines = 0
for line in f.readlines():
    #line = line[2:]
    line.rstrip()
    words = line.split('\t')
#     instance_array.append(words[0])
    sentence_array.append(words[1])
    sentiment_array.append(words[3])
    numlines = numlines + 1
print("Training set lines: ", numlines)


count = CountVectorizer(token_pattern='([^\s]{2,})', lowercase=True)
text_data = np.array(sentence_array)
bag_of_words = count.fit_transform(text_data)
X = bag_of_words.toarray()
y = np.array(sentiment_array)
clf = MultinomialNB()
start = time.time()
model = clf.fit(X, y)
model.predict(X[0:2])

numlines = 0
instance_array = []
test_array = []
f = open(sys.argv[2], "r", encoding='UTF-8')
for line in f.readlines():
    line.rstrip()
    words = line.split('\t')
    instance_array.append(words[0])
    test_array.append(words[1])
    numlines = numlines + 1
print("Test set lines: ", numlines)
# print(test_array)

i = 0
for sentence in test_array:
    test = count.transform([sentence]).toarray()
    print(instance_array[i], model.predict(test))
    i = i + 1
    
# text_data = np.array(test_array)
# bag_of_words = count.fit_transform(text_data)
# X_test = bag_of_words.toarray()
# predicted_y = model.predict(X_test[:5])
# for i, y in enumerate(predicted_y):
#     print(instance_array[i], y)

# expected results vs predicted results
# print(y_test, predicted_y)
# print("Predict:   ", model.predict_proba(X_test))
# print("Accuracy:  ", accuracy_score(y_test, predicted_y))
# print("Precision (array): ", precision_score(y_test, predicted_y, average=None))
# print("Precision (neg): ", precision_score(y_test, predicted_y, average=None))
# print("Precision (macro): ", precision_score(y_test, predicted_y, average='macro'))
# print("Recall (macro):    ", recall_score(y_test, predicted_y, average='macro'))
# print("f1 micro:  ", f1_score(y_test, predicted_y, average='micro'))
# print("f1 macro:  ", f1_score(y_test, predicted_y, average='macro'))
# print(classification_report(y_test, predicted_y))
