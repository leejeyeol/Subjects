import Preprocessing
import Methods

dataset_name = 'MLdata2.csv'

data_train, data_test, label_train, label_test = Preprocessing.get_dataset(dataset_name)

logistic_regression = Methods.logistic_regression(data_train, data_test, label_train, label_test)
naive_bayes = Methods.naive_bayes(data_train, data_test, label_train, label_test)
decision_tree = Methods.decision_tree(data_train, data_test, label_train, label_test)
random_forest = Methods.random_forest(data_train, data_test, label_train, label_test)
adaboost = Methods.adaboost(data_train, data_test, label_train, label_test)


def print_accuracy(LR, NB, DT, RF, AB):
    print("Accuracy ========================\nLogistic Regression :\t %f\nNaive Bayes :\t %f\nDecision Tree :\t %f\nRandom Forest :\t %f\nAdaBoost :\t %f\n"%(LR, NB, DT, RF, AB))


print_accuracy(logistic_regression.accuracy,naive_bayes.accuracy,decision_tree.accuracy,random_forest.accuracy,adaboost.accuracy)