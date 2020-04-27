
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
    def __init__(self, root, npoint=1024, split='train', uniform=True, normal_channel=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.normal_channel = normal_channel
        assert (split == 'train' or split == 'test')

        #

    # 根据索引获取数据的方法
    def __getitem__(self, index):
        pass

    # 获取数据的长度
    def __len__(self):
        pass

if __name__ == '__main__':
    import torch

    data = NyuHandDataLoader('../data/modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True, )
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)