#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from ELMClassifier.random_hidden_layer import RBFRandomHiddenLayer
from ELMClassifier.random_hidden_layer import SimpleRandomHiddenLayer
from ELMClassifier.elm import ELMClassifier
from ELMClassifier.random_layer import GRBFRandomLayer
import os
import sklearn.metrics as metrics
from scipy.fftpack import fft,hilbert
import time
import numpy as np
from ELMClassifier.label_smoothing_elm import ELMClassifierLabelSmooth
from config import opt
import torch
import random

def input_mapping(x,B):
    B_proj = (2. * np.pi)*B
    x = fft(x)
    xr = x.real @ np.cos(B_proj) - x.imag @ np.sin(B_proj)
    xi = x.imag @ np.cos(B_proj) + x.real @ np.sin(B_proj)
    return np.concatenate([xr,xi],axis=1)

# def input_mapping(x,B):
#     z = fft(x)
#     return np.concatenate([z.real,z.imag],axis=1)

# def input_mapping(x,B):
#     return x

def kelm_train(x_train,y_train,hidden_layer='rbf',n_hidden = 1000,use_label_smooth=False,B=None):
    print("     Begin training")
    start = time.time()
    if hidden_layer == 'rbf':
        siglayer = RBFRandomHiddenLayer(n_hidden=n_hidden, gamma=1e-3, use_exemplars=False)
    elif hidden_layer == 'sigmoid':
        siglayer = SimpleRandomHiddenLayer(n_hidden=n_hidden, activation_func='sigmoid')
    elif hidden_layer =='grbf':
        siglayer = GRBFRandomLayer(n_hidden=n_hidden, grbf_lambda=1e-3)

    if use_label_smooth:
        print("use_label_smooth:")
        clf = ELMClassifierLabelSmooth(siglayer)
    else:
        clf = ELMClassifier(siglayer)


    x_train = input_mapping(x_train,B)

    clf.fit(x_train, y_train)
    end = time.time()
    print("     Training time", end - start)
    return clf
    # joblib.dump(clf, './KELM/·' + opt.model + '_KELM_' + str(n_hidden) + '.pkl')



def kelm_test(clf ,x_test, y_test,prec1 = 0,B=None):
    print("     Begin testing")
    start = time.time()
    # print(x_test.shape)
    top1,top5 = clf.predict(input_mapping(x_test,B))

    end = time.time()
    print("     Testing time", end - start)
    # isTrue = top1 == y_test
    # # print(pre_result.size)
    # acc = np.sum(isTrue == True) / top1.size * 100

    top1acc = top1_acc(top1, y_test)
    top5acc = top5_acc(top5, y_test)
    class_acc = metrics.balanced_accuracy_score(top1, y_test) * 100
    print('     top1 Acc:',top1acc)
    print('     top5 Acc：',top5acc)
    print('     class Acc：', class_acc)
    print('     Promote',top1acc - prec1)

def top5_acc(top5,target):
    y_test_ = target.reshape(-1, 1)
    top5_is_true = y_test_ == top5
    return np.sum(top5_is_true == True) / target.size * 100

def top1_acc(top1,target):
    isTrue = top1 == target
    return np.sum(isTrue == True) / target.size * 100


def read_npys(filename):
    dict = np.load(filename,allow_pickle=True).item()
    label = dict['label']
    feature = dict['feature']
    target = dict['target']
    return label,feature,target

#DGCNN 92.22, 99.35
#PointNet 89.8703,98.7034
def get_perc():

    return 92.22, 99.35

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    # main()
    print(opt.model+'____'+opt.dataset)
    dir = 'npys/'+'DGCNNMN40'
    train_filename = dir+'/train.npy'
    test_filename = dir + '/test.npy'

    label_train,feature_train,target_train = read_npys(train_filename)
    label_test, feature_test, target_test = read_npys(test_filename)

    prec1,prec5 = get_perc()
    print('Original top1:',prec1)
    print('Original top5:',prec5)

    # label elm
    print('label based:')
    # label_clf = kelm_train(label_train,target_train,'sigmoid',1000,use_label_smooth=False)
    # kelm_test(label_clf, label_test, target_test, prec1)

    # label_clf = kelm_train(label_train, target_train, 'sigmoid', 369, use_label_smooth=False)
    # print('--------------------------------------------')
    # kelm_test(label_clf, label_test, target_test, prec1)
    #
    # Bguass = np.random.normal(loc=0.0,scale=1.0,size=(40,333))
    # label_clf = kelm_train(label_train, target_train, 'rbf', 666, B=Bguass)
    # kelm_test(label_clf, label_test, target_test, prec1,B=Bguass)

    # feature elm
    print('feature based:')
    Bguass = np.random.normal(loc=0.0,scale=1.0,size=(2048,2048))
    feature_clf =kelm_train(feature_train,target_train,'rbf',2400,B=Bguass)
    kelm_test(feature_clf,feature_test,target_test,prec1,B=Bguass)

    # rbf 2500
    # rbf 2350  -2.71
    # rbf 2400  -2.55
    # rbf 2401  -2.67
    # rbf 2500  -2.71
    # rbf 2600  -2.87
    # rbf 2700  -2.75
    # rbf 2800  -2.75
    # rbf 2900  -2.79
    # rbf 3000 -2.63
    # rbf 3100 -2.71
