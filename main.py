from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes

from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np

# Learning Models/Classifiers
clf_tree=tree.DecisionTreeClassifier()
clf_LSVC=svm.LinearSVC()
clf_SVC=svm.SVC()
clf_KNN=neighbors.KNeighborsClassifier()
clf_GNB=naive_bayes.GaussianNB()

# Data - X: Features, Y: Labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Training
clf_tree=clf_tree.fit(X,Y)
clf_LSVC=clf_LSVC.fit(X,Y)
clf_SVC=clf_SVC.fit(X,Y)
clf_KNN=clf_KNN.fit(X,Y)
clf_GNB=clf_GNB.fit(X,Y)

# prediction=clf.predict([[190,70,43]])

# Testing
prediction_tree=clf_tree.predict(X)
prediction_LSVC=clf_LSVC.predict(X)
prediction_SVC=clf_SVC.predict(X)
prediction_KNN=clf_KNN.predict(X)
prediction_GNB=clf_GNB.predict(X)


# Accuracy
a= metrics.accuracy_score(Y,prediction_tree)*100
b= metrics.accuracy_score(Y,prediction_LSVC)*100
c= metrics.accuracy_score(Y,prediction_SVC)*100
d= metrics.accuracy_score(Y,prediction_KNN)*100
e= metrics.accuracy_score(Y,prediction_GNB)*100

x=('Tree','LinearSVC','SVC','KNN','GaussianNB')
ind=np.arange(len(x))

y=[a, b, c, d, e]

fig, ax = plt.subplots()

ax.barh(ind,y,align='center')
ax.set_yticks(ind)
ax.set_yticklabels(x)
ax.set_xlabel('Accuracy')
ax.set_title('Classification accuracy comparison')

plt.show()