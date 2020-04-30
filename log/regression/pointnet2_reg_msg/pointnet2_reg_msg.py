import torch
import torch.nn as nn
import torch.nn.functional as F # 模块中定义的常用函数和类
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction




class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        # super(get_model,self) 首先找到 get_model 的父类（就是类 nn.Module），然后把类 get_model 的对象转换为类 nn.Module 的对象
        super(get_model, self).__init__()
        # 如果增加法向量信息，则 in_channel = 3
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # 有三层特征提取
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        # 有三个全连接层，最后输出指定维数的特征向量
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)         # xyz 是降采样的点，points 是这些点从低维映射到高维以后的
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.mse_loss(pred, target)
        #total_loss = F.nll_loss(pred, target)   # 输入 是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
                                                # 适合最后一层是log_softmax()的网络

        return total_loss




if __name__ == '__main__':
    x = get_model(10, normal_channel=True)
    print(x)