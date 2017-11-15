# import packages
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

def make_data():
    x_1 = np.random.binomial(1, 0.5)
    x_2 = np.random.binomial(1, 0.5)
    x_3 = np.random.uniform(-2, 2)
    x_4_star = np.random.triangular(-2, 0, 2)
    x_4 = (1/3)*x_3 + (2/3)*x_4_star
    epsilon = np.random.normal(0, (0.5*0.5))
    y = (-1*x_1) + (2*x_2) - 0.5*np.square((1/np.sqrt(2)*x_3)+(1/np.sqrt(2)*x_4)+(1/np.sqrt(2))) + 6 + epsilon
    return [y, x_1, x_2, x_3, x_4]


def make_dataset():
    dataset = []
    for i in range(0, 100):
        dataset.append(make_data())
    return np.asarray(dataset)


def get_dataset():
    dataset = make_dataset()
    data_train, data_test, label_train, label_test = train_test_split(dataset[:,1:], dataset[:,0], train_size=50)
    return data_train, data_test, label_train, label_test

def get_MSE(pred, true):
    mse = (np.square(pred - true)).mean(axis=0)
    return mse

def get_MAE(pred, true):
    mae = (np.abs(pred - true)).mean(axis=0)
    return mae


MSE_NN = []
MAE_NN = []
MSE_RF = []
MAE_RF = []
for i in range(1, 50):
    data_train, data_test, label_train, label_test = get_dataset()
    reg = MLPRegressor(hidden_layer_sizes=(10, 5), activation = 'relu', solver='adam', batch_size='auto', shuffle=True)
    reg = reg.fit(data_train, label_train)
    result = reg.predict(data_test)
    MSE_NN.append(get_MSE(result, label_test))
    MAE_NN.append(get_MAE(result, label_test))

    reg = RandomForestRegressor(max_depth=5, random_state=0)
    reg = reg.fit(data_train, label_train)
    result = reg.predict(data_test)
    MSE_RF.append(get_MSE(result, label_test))
    MAE_RF.append(get_MAE(result, label_test))


print("MAE_NN_mean: "+str(np.mean(MAE_NN)))
print("MAE_NN_var : "+str(np.var(MAE_NN)))
print("MSE_NN_mean: "+str(np.mean(MSE_NN)))
print("MSE_NN_var : "+str(np.var(MSE_NN)))
print("\n")
print("MAE_RF_mean: "+str(np.mean(MAE_RF)))
print("MAE_RF_var : "+str(np.var(MAE_RF)))
print("MSE_RF_mean: "+str(np.mean(MSE_RF)))
print("MSE_RF_var : "+str(np.var(MSE_RF)))

