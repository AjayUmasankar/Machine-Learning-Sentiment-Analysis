# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
'''
Created on 28 Jul 2019

@author: Ajay
'''
from preprocess import preprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
import time

# DT uses max_features=200 in addition to normal CountVectorizer arguments
pp = preprocessor(1500, "sentiment", "dt")
X_train = pp.X_train
X_test = pp.X_test
y_train = pp.y_train
y_test = pp.y_test

# if random_state is not set, the features are randomised, therefore the tree may be different each time
clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0, min_samples_leaf=20)
start = time.time()
model = clf.fit(X_train, y_train)
stop = time.time()

predicted_y = model.predict(X_test)

# expected results vs predicted results
# print(y_test, predicted_y)
# print("Predict:   ", model.predict_proba(X_test))
print("Accuracy:  ", accuracy_score(y_test, predicted_y))
print("Precision (array): ", precision_score(y_test, predicted_y, average=None))
# print("Precision (neg): ", precision_score(y_test, predicted_y, average=None))
print("Precision (macro): ", precision_score(y_test, predicted_y, average='macro'))
print("Recall (macro):    ", recall_score(y_test, predicted_y, average='macro'))
# print("f1 micro:  ", f1_score(y_test, predicted_y, average='micro'))
# print("f1 macro:  ", f1_score(y_test, predicted_y, average='macro'))
print(classification_report(y_test, predicted_y))
print('Time: ', stop - start) 
#