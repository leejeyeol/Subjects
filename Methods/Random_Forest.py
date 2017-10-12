import Preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data_train, data_test, label_train, label_test = Preprocessing.get_dataset('../MLdata2.csv')
RandomForest = RandomForestClassifier(max_depth=2, random_state=72170300)
RandomForest.fit(data_train, label_train)

result = RandomForest.predict(data_test)
accuracy_score(label_test, result)
print("Random Forest accuracy : %f" % accuracy_score(label_test, result))