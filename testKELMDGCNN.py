# -*- coding: utf-8 -*-
import pickle
from ELMClassifier.random_hidden_layer import RBFRandomHiddenLayer,SimpleRandomHiddenLayer
from ELMClassifier.random_layer import GRBFRandomLayer
from ELMClassifier.elm import ELMClassifier
import sklearn.metrics as metrics
import time
import numpy as np
from ELMClassifier.label_smoothing_elm import ELMClassifierLabelSmooth
from config import opt
import random

def kelm_train(x_train,y_train,hidden_layer='rbf',n_hidden = 1000,use_label_smooth=True):
    print("     Begin Training：")
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
    clf.fit(x_train, y_train)
    end = time.time()
    print("     Training time", end - start)
    return clf


def kelm_test(clf ,x_test, y_test,prec1 = 0):
    print("     Begin Testing：")
    start = time.time()
    # print(x_test.shape)
    top1,top5 = clf.predict(x_test)



    end = time.time()
    print("     Testing time", end - start)
    # isTrue = top1 == y_test
    # # print(pre_result.size)
    # acc = np.sum(isTrue == True) / top1.size * 100

    top1acc = top1_acc(top1, y_test)
    top5acc = top5_acc(top5, y_test)
    class_acc = metrics.balanced_accuracy_score(top1, y_test) * 100
    print('     top1:',top1acc)
    print('     top5：',top5acc)
    print('     clsacc：', class_acc)
    print('     Distance',top1acc - prec1)

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


def get_perc():

    return 92.22, 99.35

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    # Get features from random initilized DGCNN on ModelNet40
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
    #  label feature KELM:
    label_clf = kelm_train(label_train, target_train, 'sigmoid', 257, use_label_smooth=False)
    print('--------------------------------------------')
    kelm_test(label_clf, label_test, target_test, prec1)

    label_clf = kelm_train(label_train, target_train, 'rbf', 333, use_label_smooth=False)
    kelm_test(label_clf, label_test, target_test, prec1)

    # 2048 global feature KELM (Paper):
    print('feature based:')
    feature_clf =kelm_train(feature_train,target_train,'rbf',2048,use_label_smooth=False)
    kelm_test(feature_clf,feature_test,target_test,prec1)