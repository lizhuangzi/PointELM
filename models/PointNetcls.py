import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.PointNet import PointNetEncoder, feature_transform_reguliarzer

class PoiintNet(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(PoiintNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        gbfeature, trans, trans_feat,maxbefore_feaute = self.feat(x)
        meanfeature = torch.mean(maxbefore_feaute, 2)
        newgbfeature  = torch.cat([gbfeature,meanfeature],dim=1)

        x = self.fc1(gbfeature)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.bn2(x)

        x = F.relu(x)
        cls_before_feaute = x
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        return x, newgbfeature,cls_before_feaute

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__=="__main__":

    params={
        "up_ratio":4,
        "patch_num_point":100
    }
    clsnet=PoiintNet(10,normal_channel=False).cuda()
    point_cloud=torch.rand(2,3,1024).cuda()
    output,_=clsnet(point_cloud)
    print(output.shape)