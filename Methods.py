from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

class classify_method:
    def __init__(self, data_train, data_test, label_train, label_test):
        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train
        self.label_test = label_test

class logistic_regression(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy, self.confusion_matrix, self.auc = self.run()

    def run(self):
        LogisticRegression = linear_model.LogisticRegression(C=1e5)
        LogisticRegression.fit(self.data_train, self.label_train)
        result = LogisticRegression.predict(self.data_test)

        fpr, tpr, thresholds = metrics.roc_curve(self.label_test, result, pos_label=2)
        auc=metrics.auc(fpr, tpr)

        return result, accuracy_score(self.label_test, result), confusion_matrix(self.label_test, result), auc

class deep_neural_networks(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy, self.confusion_matrix, self.auc = self.run()

    def run(self):
        MLP = MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), max_iter=1000, activation='relu',
                    solver='adam', random_state=1,
                    learning_rate_init=0.02, learning_rate='constant')
        MLP.fit(self.data_train, self.label_train)
        result = MLP.predict(self.data_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.label_test, result, pos_label=2)
        auc = metrics.auc(fpr, tpr)
        print(MLP.n_layers_)

        return result, accuracy_score(self.label_test, result), confusion_matrix(self.label_test, result), auc



class naive_bayes(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy, self.confusion_matrix, self.auc = self.run()

    def run(self):
        NaiveBayes = GaussianNB()
        NaiveBayes.fit(self.data_train, self.label_train)
        result = NaiveBayes.predict(self.data_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.label_test, result, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        return result, accuracy_score(self.label_test, result), confusion_matrix(self.label_test, result), auc


class decision_tree(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy, self.confusion_matrix, self.auc = self.run()

    def run(self):
        DecisionTree = tree.DecisionTreeClassifier()
        DecisionTree.fit(self.data_train, self.label_train)
        result = DecisionTree.predict(self.data_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.label_test, result, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        return result, accuracy_score(self.label_test, result), confusion_matrix(self.label_test, result), auc


class random_forest(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy, self.confusion_matrix, self.auc = self.run()

    def run(self):
        RandomForest = RandomForestClassifier(max_depth=2, random_state=72170300)
        RandomForest.fit(self.data_train, self.label_train)
        result = RandomForest.predict(self.data_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.label_test, result, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        return result, accuracy_score(self.label_test, result), confusion_matrix(self.label_test, result), auc

class adaboost(classify_method):
    def __init__(self, data_train, data_test, label_train, label_test):
        super().__init__(data_train, data_test, label_train, label_test)
        self.predicted_result, self.accuracy, self.confusion_matrix, self.auc = self.run()

    def run(self):
        Adaboost = AdaBoostClassifier()
        Adaboost.fit(self.data_train, self.label_train)
        result = Adaboost.predict(self.data_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.label_test, result, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        return result, accuracy_score(self.label_test, result), confusion_matrix(self.label_test, result), auc

