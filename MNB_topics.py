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
from sklearn.naive_bayes import MultinomialNB
#GOAL=0.74

pp = preprocessor("topic", "mnb")

clf = MultinomialNB()
model = clf.fit(pp.X_train, pp.y_train)

predicted_y = model.predict(pp.X_test)
i = pp.divider
for y in predicted_y:
    print(pp.instance_array[i], y)
    i = i + 1

