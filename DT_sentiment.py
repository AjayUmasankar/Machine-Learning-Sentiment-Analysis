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
from sklearn import tree

#GOAL=0.74

pp = preprocessor("sentiment", "dt")

clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0, min_samples_leaf=20)
model = clf.fit(pp.X_train, pp.y_train)

predicted_y = model.predict(pp.X_test)
i = pp.divider
for y in predicted_y:
    print(pp.instance_array[i], y)
    i = i + 1
