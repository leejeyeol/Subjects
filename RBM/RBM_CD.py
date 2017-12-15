#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)
 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials
"""

import sys
import numpy
from sklearn import metrics

numpy.seterr(all='ignore')

def list_to_one_hot(list_):
    one_hot_list = []
    for element in range(len(list_)):
        one_hot_list.append([])
        for cls in list(set(list_)):
            if list_[element] == cls:
                one_hot_list[element].append(1)
            else:
                one_hot_list[element].append(0)
    return one_hot_list


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


class RBM(object):
    def __init__(self, input=None, label=None, n_visible=4, n_hidden=4, n_label=3, momentum_coefficient=0.5,\
                 W=None, U=None, hbias=None, vbias=None, lbias=None, numpy_rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer
        self.n_label = n_label # num of labels in output layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(72170300)

        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W
        if U is None:
            a = 1./ n_label
            initial_U = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_label, n_hidden)))

            U = initial_U

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0

        if lbias is None:
            lbias = numpy.zeros(n_label)  # initialize l bias 0

        self.momentum_coefficient = momentum_coefficient
        self.numpy_rng = numpy_rng
        self.input = input
        self.input_std = numpy.std(input)
        self.label = label
        self.W = W
        self.U = U
        self.hbias = hbias
        self.vbias = vbias
        self.lbias = lbias

        self.Wm = numpy.zeros([n_visible,n_hidden])
        self.Um = numpy.zeros([n_label,n_hidden])
        self.hbm = numpy.zeros(n_hidden)
        self.vbm = numpy.zeros(n_visible)
        self.lbm = numpy.zeros(n_label)


        # self.params = [self.W, self.U self.hbias, self.vbias, self.lbias]

    def contrastive_divergence(self, lr=0.1, k=1, input=None, label=None):
        if input is not None:
            self.input = input
        if label is not None:
            self.label = label

        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_vy(self.input, self.label)

        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples, \
                nh_means, nh_samples, \
                nl_means, nl_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, \
                nh_means, nh_samples, \
                nl_means, nl_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples

        self.Wm *= self.momentum_coefficient
        self.Um *= self.momentum_coefficient
        self.hbm *= self.momentum_coefficient
        self.vbm *= self.momentum_coefficient
        self.lbm *= self.momentum_coefficient

        self.Wm += (numpy.dot(self.input.T/self.input_std, ph_mean) - numpy.dot(nv_samples.T/self.input_std, nh_means))
        self.Um += (numpy.dot(self.label.T, ph_mean) - numpy.dot(nl_samples.T, nh_means))
        self.vbm += numpy.mean(self.input/self.input_std - nv_samples/self.input_std, axis=0)
        self.hbm += numpy.mean(ph_sample - nh_means, axis=0)
        self.lbm += numpy.mean(self.label - nl_samples, axis=0)


        self.W += lr * self.Wm
        self.U += lr * self.Um
        self.vbias += lr * self.vbm
        self.hbias += lr * self.hbm
        self.lbias += lr * self.lbm



        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    def sample_h_given_vy(self, v0_sample,l0_sample):
        h1_mean = self.propup(v0_sample, l0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,  # discrete: binomial
                                            n=1,
                                            p=h1_mean)

        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.normal(v1_mean, self.input_std, v1_mean.shape)

        return [v1_mean, v1_sample]

    def sample_l_given_h(self, h0_sample):
        l1_mean = self.propdown_label(h0_sample)
        prob=numpy.exp(numpy.mean(numpy.dot(h0_sample, self.U.T) + self.lbias, axis=0)) / sum(numpy.exp(numpy.mean(numpy.dot(h0_sample, self.U.T) + self.lbias, axis=0)))
        l1_sample = self.numpy_rng.multinomial(1, prob,size=self.label.shape[0])
        return [l1_mean, l1_sample]

    def propup(self, v, l):
        pre_sigmoid_activation = numpy.dot(v/self.input_std, self.W) + numpy.dot(l, self.U) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)

    def propdown_label(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.U.T) + self.lbias
        return sigmoid(pre_sigmoid_activation)

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        l1_mean, l1_sample = self.sample_l_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_vy(v1_sample, l1_sample)


        return [v1_mean, v1_sample,
                h1_mean, h1_sample,
                l1_mean, l1_sample]

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy = - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
                      (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

    def get_accuracy(self, input, label):
        h = sigmoid(numpy.dot(input, self.W) + self.hbias)
        y = sigmoid(numpy.dot(h, self.U.T) + self.lbias)

        O = 0
        X = 0

        for i in range(len(y)):
            y[i] = numpy.int32(y[i] == y[i].max())

            if numpy.all(y[i] == label[i]):
                O += 1
            else:
                X += 1
        print("accuracy : %f" % (O / O + X))

        y_int = [numpy.where(r == 1)[0][0] for r in y]
        label_int = [numpy.where(r == 1)[0][0] for r in label]
        fpr, tpr ,_ = metrics.roc_curve(label_int, y_int, pos_label=1)

        print("auc : %f" % (metrics.auc(fpr, tpr)))


def test_rbm(dataname="new-thyroid.data", learning_rate=0.01, k=1, training_epochs=1000):

    data = numpy.loadtxt(dataname, delimiter=',')
    numpy.random.shuffle(data)
    data = numpy.int32(data)
    if dataname == "new-thyroid.data":
        label = data[:, 0]
        label = list_to_one_hot(label)
        data = data[:, 1:-1]
        data = (data - data.min(0)) / (data - data.min(0)).ptp(0)

        train_data = data[0:142, :]
        train_label = label[0:142]
        train_label = numpy.asarray(train_label)
        test_data = data[143:-1,:]
        test_label = label[143:-1]
        test_label = numpy.asarray(test_label)

    if dataname == "pima-indians-diabetes.data":
        label = data[:, -1]
        label = list_to_one_hot(label)
        data = data[:, 0:-2]
        data = (data - data.min(0)) / (data - data.min(0)).ptp(0)

        train_data = data[0:512, :]
        train_label = label[0:512]
        train_label = numpy.asarray(train_label)
        test_data = data[513:-1, :]
        test_label = label[513:-1]
        test_label = numpy.asarray(test_label)

    rng = numpy.random.RandomState(72170300)

    # construct RBM
    rbm = RBM(input=train_data, label=train_label, numpy_rng=rng, n_visible=train_data.shape[1], n_label=train_label.shape[1])

    # train
    for epoch in range(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print('Training epoch %d, cost is %f' % (epoch, cost))

    # test
    '''
    v = numpy.array([[0, 0, 0, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0]])
    '''
    rbm.get_accuracy(test_data, test_label)
    #print(rbm.reconstruct(test_data))
#    print(rbm.get_accuracy(train_data,train_label))

if __name__ == "__main__":
    test_rbm()