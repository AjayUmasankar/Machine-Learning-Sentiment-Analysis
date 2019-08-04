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
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#################### PREPROCESSING ENDS HERE #####################
pp = preprocessor(1500, "topic", "bnb")
X_train = pp.X_train
X_test = pp.X_test
y_train = pp.y_train
y_test = pp.y_test

clf = BernoulliNB()
model = clf.fit(X_train, y_train)

predicted_y = model.predict(X_test)

# expected results vs predicted results
#print(y_test, predicted_y)
print("Predict:   ", model.predict_proba(X_test))
print("Accuracy:  ", accuracy_score(y_test, predicted_y))
print("Precision: ", precision_score(y_test, predicted_y, average='micro'))
print("Recall:    ", recall_score(y_test, predicted_y, average='micro'))
print("Precision: ", precision_score(y_test, predicted_y, average='macro'))
print("Recall:    ", recall_score(y_test, predicted_y, average='macro'))
# print("f1 micro:  ", f1_score(y_test, predicted_y, average='micro'))
# print("f1 macro:  ", f1_score(y_test, predicted_y, average='macro'))
print(classification_report(y_test, predicted_y))