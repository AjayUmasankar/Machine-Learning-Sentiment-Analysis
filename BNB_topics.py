# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# 
# '''
# Created on 26 Jul 2019
# 
# @author: Ajay
# '''
# 
from pp_final import preprocessor
from sklearn.naive_bayes import BernoulliNB

pp = preprocessor("topic", "bnb")

clf = BernoulliNB()
model = clf.fit(pp.X_train, pp.y_train)

predicted_y = model.predict(pp.X_test)
i = pp.divider
for y in predicted_y:
    print(pp.instance_array[i], y)
    i = i + 1
# i = 0
# for sentence in test_array:
#     test = count.transform([sentence]).toarray()
#     print(instance_array[i], model.predict(test))
#     i = i + 1
    
# text_data = np.array(test_array)
# bag_of_words = count.fit_transform(text_data)
# X_test = bag_of_words.toarray()
# predicted_y = model.predict(X_test[:5])
# for i, y in enumerate(predicted_y):
#     print(instance_array[i], y)
