import Preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

data_train, data_test, label_train, label_test = Preprocessing.get_dataset('../MLdata2.csv')
Adaboost = AdaBoostClassifier()
Adaboost.fit(data_train, label_train)

result = Adaboost.predict(data_test)
accuracy_score(label_test, result)
print("Adaboost accuracy : %f" % accuracy_score(label_test, result))