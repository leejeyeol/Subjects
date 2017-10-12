import Preprocessing
from sklearn.metrics import accuracy_score
from sklearn import tree

data_train, data_test, label_train, label_test = Preprocessing.get_dataset('../MLdata2.csv')
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree.fit(data_train, label_train)

result = DecisionTree.predict(data_test)
accuracy_score(label_test, result)
print("Decision Tree accuracy : %f" % accuracy_score(label_test, result))