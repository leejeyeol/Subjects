from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

class classify_method:
    def __init__(self, data_train, data_test, label_train, label_test):
        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train
        self.label_test = label_test

class logistic_regression(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy = self.run()

    def run(self):
        LogisticRegression = linear_model.LogisticRegression(C=1e5)
        LogisticRegression.fit(self.data_train, self.label_train)
        result = LogisticRegression.predict(self.data_test)

        return result, accuracy_score(self.label_test, result)


class naive_bayes(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy = self.run()

    def run(self):
        NaiveBayes = GaussianNB()
        NaiveBayes.fit(self.data_train, self.label_train)
        result = NaiveBayes.predict(self.data_test)

        return result, accuracy_score(self.label_test, result)


class decision_tree(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy = self.run()

    def run(self):
        DecisionTree = tree.DecisionTreeClassifier()
        DecisionTree.fit(self.data_train, self.label_train)
        result = DecisionTree.predict(self.data_test)

        return result, accuracy_score(self.label_test, result)


class random_forest(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy = self.run()

    def run(self):
        RandomForest = RandomForestClassifier(max_depth=2, random_state=72170300)
        RandomForest.fit(self.data_train, self.label_train)
        result = RandomForest.predict(self.data_test)

        return result, accuracy_score(self.label_test, result)

class adaboost(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy = self.run()

    def run(self):
        Adaboost = AdaBoostClassifier()
        Adaboost.fit(self.data_train, self.label_train)
        result = Adaboost.predict(self.data_test)

        return result, accuracy_score(self.label_test, result)

