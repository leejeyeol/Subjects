import Preprocessing
from sklearn.metrics import accuracy_score
from sklearn import linear_model

data_train, data_test, label_train, label_test = Preprocessing.get_dataset('../MLdata2.csv')
LogisticRegression = linear_model.LogisticRegression(C=1e5)
LogisticRegression.fit(data_train, label_train)

result = LogisticRegression.predict(data_test)
accuracy_score(label_test, result)
print("Logistic Regrresion accuracy : %f" % accuracy_score(label_test, result))