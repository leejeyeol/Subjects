import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import Methods


def detach_label_and_data(data):
    label = data[:, -1]
    data = data[:, :-1]

    return data, label

def get_dataset():
    data = genfromtxt('CTG_raw_data.csv', delimiter=',', skip_header=1)
    data, label = detach_label_and_data(data)
    data_train, data_test, label_train, label_test = train_test_split(data, label, train_size=0.7,
                                                                      random_state=72170300)
    return data_train, data_test, label_train, label_test


data_train, data_test, label_train, label_test = get_dataset()

decision_tree = Methods.decision_tree(data_train, data_test, label_train, label_test)
random_forest = Methods.random_forest(data_train, data_test, label_train, label_test)
naive_bayes = Methods.naive_bayes(data_train, data_test, label_train, label_test)
deep_neural_networks = Methods.deep_neural_networks(data_train, data_test, label_train, label_test)

print("\nDT_accuracy :" + str(decision_tree.accuracy)+"\tDT_auc :"+str(decision_tree.auc))
print(decision_tree.confusion_matrix)
print("\nRF_accuracy :" + str(random_forest.accuracy)+"\tRF_auc :"+str(random_forest.auc))
print(random_forest.confusion_matrix)
print("\nNB_accuracy :" + str(naive_bayes.accuracy)+"\tNB_auc :"+str(naive_bayes.auc))
print(naive_bayes.confusion_matrix)
print("\nDNN_accuracy :" + str(deep_neural_networks.accuracy)+"\tDNN_auc :"+str(deep_neural_networks.auc))
print(deep_neural_networks.confusion_matrix)