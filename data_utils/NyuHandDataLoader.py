import numpy as np
import os
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point



# Dataset 是抽象类，不能实例化，只能由其子类继承。记住就行了
class NyuHandDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=True, normal_channel=False, cache_size=1000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.normal_channel = normal_channel

        # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        assert (split == 'train_body' or split == 'test_body' or split == 'train_hand' or split == 'test_hand')

        # 从标签文件，提取每张图的信息，从 hand_ids_coord 提取每张图的编号 hand_ids 和 坐标信息 hand_coord
        hand_ids_coord = [line.rstrip().split(' ') for line in open(os.path.join(self.root, split, 'hand_joint_label.txt'))]
        hand_ids = [ ''.join(x[0:1]) for x in hand_ids_coord]
        hand_coord = [ x[1:]  for x in hand_ids_coord]

        # 将图片编号，图片路径，手关节坐标，列为一个元组
        self.hand_all = [(i, os.path.join(self.root, split , 'cloudpoints/', hand_ids[i] + '.txt'), hand_coord[i]) for
                         i in range(len(hand_ids))]
        print('The size of %s data is %d' % (split, len(self.hand_all)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        index = 1
        hand_input = np.loadtxt(self.hand_all[index][1]).astype(np.float32)  # 读取点云txt文件，并转换为numpy
        hand_joint = np.array(self.hand_all[index][2]).astype(np.float32)  # 把关节点的坐标转换成 numpy 格式

        hand_input[:, 0:3] = pc_normalize(hand_input[:, 0:3])




    # 根据索引获取数据的方法
    def __getitem__(self, index):
        if index in self.cache:
            hand_input, hand_joint = self.cache[index]
        else:
            hand_all_index = self.hand_all[index]
            assert(hand_all_index[0] == index)
            # 读取完整的点云坐标文件
            hand_input = np.loadtxt(self.hand_all[index][1]).astype(np.float32)  # 读取点云txt文件，并转换为numpy
            hand_joint = np.array(self.hand_all[index][2]).astype(np.float32)  # 把关节点的坐标转换成 numpy 格式
            if self.uniform:
                hand_input = farthest_point_sample(hand_input, self.npoints)
            else:
                hand_input = hand_input[0:self.npoints, :]
            hand_input[:, 0:3] = pc_normalize(hand_input[:, 0:3])

            if self.normal_channel:
                pass

            if len(self.cache) < self.cache_size:
                self.cache[index] = (hand_input, hand_joint)




        return hand_input, hand_joint


    # 获取数据的长度
    def __len__(self):
        return len(self.hand_all)

if __name__ == '__main__':
    import torch

    data = NyuHandDataLoader('../data/nyu_hand_dataset_v2/', split='train_hand', uniform=True, normal_channel=False, )
    DataLoader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)