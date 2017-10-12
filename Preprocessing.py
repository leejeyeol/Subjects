# import packages
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

# load dataset
def remove_NAN(data):
    nan_indexes = []
    for i in range(0, data.shape[0]):
        if np.isnan(data[i]).any():
            nan_indexes.append(i)
    nan_indexes.sort(reverse=True)

    for j in range(0, len(nan_indexes)):
        data = np.delete(data, nan_indexes[j], 0)
    return data

def detach_label_and_data(data):
    label = data[:, 59]
    data = data[:, :59]

    return data, label

def get_dataset(file_name):
    # 'MLdata2.csv'
    data = genfromtxt(file_name, delimiter=',')
    data = remove_NAN(data)
    data, label = detach_label_and_data(data)
    data_train, data_test, label_train, label_test = train_test_split(data, label, train_size=0.66,
                                                                      random_state=72170300)
    return data_train, data_test, label_train, label_test





