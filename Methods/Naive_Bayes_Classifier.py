import Preprocessing
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

data_train, data_test, label_train, label_test = Preprocessing.get_dataset('../MLdata2.csv')
NaiveBayes = GaussianNB()
NaiveBayes.fit(data_train, label_train)

result = NaiveBayes.predict(data_test)
accuracy_score(label_test, result)
print("Naive Bayes accuracy : %f" % accuracy_score(label_test, result))