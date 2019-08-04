# Load libraries
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Brazil is best',
                      'I like Germany more, Germany beats both',
                      'I like Italy, because Italy is beautiful',
                      'I am from Germany, so I like Germany more'])
    
text_data = np.array(text_data)


# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
print(count.get_feature_names())

# Create feature matrix
X = bag_of_words.toarray()

# Create target vector
y = np.array([0,0,1,0,1])

X_train = X[:3]
X_test = X[3:]
y_train = y[:3]
y_test = y[3:]

# Create new unseen test instances
test1 = count.transform(['I like Italy, because Italy is beautiful']).toarray()
test2 = count.transform(['I am from Germany, so I like Germany more']).toarray()

print("----bnb")
clf = BernoulliNB()
model = clf.fit(X_train, y_train)

print(model.predict(test1))
print(model.predict(test2))
predicted_y = model.predict(X_test)
#expected results vs predicted results
print(y_test, predicted_y)
print("Predict:   ", model.predict_proba(X_test))
print("Accuracy:  ", accuracy_score(y_test, predicted_y))
print("Precision: ", precision_score(y_test, predicted_y))
print("Recall:    ", recall_score(y_test, predicted_y))
print("f1 micro:  ", f1_score(y_test, predicted_y, average='micro'))
print("f1 macro:  ", f1_score(y_test, predicted_y, average='macro'))
print(classification_report(y_test, predicted_y))

# print("----mnb")
# clf = MultinomialNB()
# model = clf.fit(X_train, y_train)
# 
# print(model.predict(test1))
# print(model.predict(test2))
# predicted_y = model.predict(X_test)
# print(y_test, predicted_y)
# print(model.predict_proba(X_test))
# print(accuracy_score(y_test, predicted_y))
# print(precision_score(y_test, predicted_y))
# print(recall_score(y_test, predicted_y))
# print(f1_score(y_test, predicted_y, average='micro'))
# print(f1_score(y_test, predicted_y, average='macro'))
# print(classification_report(y_test, predicted_y))
# 
# # if random_state is not set, the features are randomised, therefore the tree may be different each time
# print("----dt")
# clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
# model = clf.fit(X_train, y_train)
# 
# print(model.predict(test1))
# print(model.predict(test2))
# predicted_y = model.predict(X_test)
# print(y_test, predicted_y)
# print(model.predict_proba(X_test))
# print(accuracy_score(y_test, predicted_y))
# print(precision_score(y_test, predicted_y))
# print(recall_score(y_test, predicted_y))
# print(f1_score(y_test, predicted_y, average='micro'))
# print(f1_score(y_test, predicted_y, average='macro'))
# print(classification_report(y_test, predicted_y))