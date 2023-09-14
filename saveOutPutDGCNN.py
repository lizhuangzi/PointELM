import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from models.DGCNNcls import DGCNN_cls
from data.ModelNet40 import ModelNet40

import random
import argparse

def save_npy(dict,dir,is_train = True):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if is_train:
        np.save(dir+'/train.npy',dict)
    else:
        np.save(dir+'/test.npy',dict)

def infer(model, data_loader, is_train = True):
    model.eval()
    total_number = 0
    train_label_output = []
    train_feature_output = []
    train_target = []
    train_istrue = []
    with torch.no_grad():
        for i, (data,_, label) in enumerate(tqdm(data_loader)):
            batch_size = data.shape[0]
            data, label = data.cuda(), label.cuda().squeeze()
            data = data.permute(0, 2, 1)

            logits,gbf,_ = model(data)

            pred = logits.max(dim=1)[1]
            #logits = F.tanh(logits)
            isTrue = pred.cpu().numpy() == label.cpu().numpy()


            if len(train_label_output) == 0:
                train_label_output = logits.cpu().numpy()
                train_feature_output = gbf.cpu().numpy()
                train_target = label.cpu().numpy()
                train_istrue = isTrue
            else:
                train_label_output = np.concatenate((train_label_output, logits.cpu().numpy()), axis=0)
                train_feature_output = np.concatenate((train_feature_output, gbf.cpu().numpy()), axis=0)
                train_target = np.concatenate((train_target, label.cpu().numpy()), axis=0)
                train_istrue = np.concatenate((train_istrue, isTrue), axis=0)

            total_number += batch_size

        # if is_train:
        #     train_label_output = train_label_output[train_istrue]
        #     train_feature_output = train_feature_output[train_istrue]
        #     train_target = train_target[train_istrue]

        feature_dict = {}
        print(train_label_output.shape)
        print(train_feature_output.shape)
        print(train_target.shape)
        print(train_istrue)

        feature_dict['label'] = train_label_output
        feature_dict['feature'] = train_feature_output
        feature_dict['target'] = train_target
        feature_dict['is_true'] = train_istrue

        save_npy(feature_dict, 'npys/' + 'DGCNNMN40', is_train)
        print('----------end-------------')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()

    seed = 1234
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    # data_path = '/home/lizhuangzi/Desktop/MetaSampler-main/data/modelnet40_normal_resampled/'
    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True,
    #                                               num_workers=10, drop_last=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False,
    #                                              num_workers=10)

    trainSet = ModelNet40(data_dir='./data', partition='train', num_points=32)
    testSet = ModelNet40(data_dir='./data', partition='test', num_points=32)
    train_loader = DataLoader(trainSet, batch_size=32,
                                 shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(testSet, batch_size=32,
                                 shuffle=False, pin_memory=True, num_workers=4)

    n_classes = 40


    model = DGCNN_cls(n_classes)
    model.load_state_dict(torch.load('./PretrainModel/'+'DGCN_ModelNet40.parm'))
    model = model.cuda()

    # model = Pct(args).cuda()
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('../PretrainModel/'+'PCTNetMN40.parm'))
    # model = model.cuda()
    # model.eval()
    infer(model, train_loader, is_train=True)
    infer(model, test_loader, is_train=False)



