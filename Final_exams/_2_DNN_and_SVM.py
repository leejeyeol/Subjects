from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np

def MAE(S,S_):
    return np.sum(np.abs(S-S_))/len(S)
def RMSE(S,S_):
    return np.sqrt(np.sum(np.square(S-S_))/len(S))
def SVRMSE(S,S_):
    return RMSE(S,S_)/(np.sum(S)/len(S))

number_of_folds = 10
train_sets = []
test_sets = []
train_labels = []
test_labels = []
SVM_results = []
DNN_results = []
SVM_MAEs = []
SVM_RMSEs = []
SVM_SVRMSEs = []
DNN_MAEs = []
DNN_RMSEs = []
DNN_SVRMSEs = []

mins = np.genfromtxt('KospiData_min_max.csv', delimiter=',')[1:,1:][0]
maxes = np.genfromtxt('KospiData_min_max.csv', delimiter=',')[1:,1:][1]

for i in range(number_of_folds):
    train_data = np.genfromtxt('cv%d_tr.csv' % (i + 1), delimiter=',')[1:, 1:]
    test_data = np.genfromtxt('cv%d_te.csv'%(i+1), delimiter=',')[1:, 1:]

    train_sets.append((train_data[:, 1:]-mins[1:])/(maxes[1:]-mins[1:]))
    test_sets.append((test_data[:, 1:]-mins[1:])/(maxes[1:]-mins[1:]))
    train_labels.append((train_data[:, 0]-mins[0])/(maxes[0]-mins[0]))
    test_labels.append((test_data[:, 0]-mins[0])/(maxes[0]-mins[0]))

for i in range(number_of_folds):
    clf = SVR()
    clf.fit(train_sets[i], train_labels[i])
    SVM_results.append(clf.predict(test_sets[i]))

    MLP = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50), max_iter=1000, activation='relu',
                    solver='adam', random_state=1,
                    learning_rate_init=0.02, learning_rate='constant')
    MLP.fit(train_sets[i], train_labels[i])
    DNN_results.append(MLP.predict(test_sets[i]))

    SVM_results[i] = SVM_results[i] * (maxes[0]-mins[0])+mins[0]
    DNN_results[i] = DNN_results[i] * (maxes[0]-mins[0])+mins[0]
    test_labels[i] = test_labels[i] * (maxes[0]-mins[0])+mins[0]

    SVM_MAEs.append(MAE(test_labels[i], SVM_results[i]))
    SVM_RMSEs.append(RMSE(test_labels[i], SVM_results[i]))
    SVM_SVRMSEs.append(SVRMSE(test_labels[i], SVM_results[i]))
    DNN_MAEs.append(MAE(test_labels[i], DNN_results[i]))
    DNN_RMSEs.append(RMSE(test_labels[i], DNN_results[i]))
    DNN_SVRMSEs.append(SVRMSE(test_labels[i], DNN_results[i]))

print(np.mean(SVM_MAEs))
print(np.mean(SVM_RMSEs))
print(np.mean(SVM_SVRMSEs))
print(np.mean(DNN_MAEs))
print(np.mean(DNN_RMSEs))
print(np.mean(DNN_SVRMSEs))

print(np.std(SVM_MAEs))
print(np.std(SVM_RMSEs))
print(np.std(SVM_SVRMSEs))
print(np.std(DNN_MAEs))
print(np.std(DNN_RMSEs))
print(np.std(DNN_SVRMSEs))
