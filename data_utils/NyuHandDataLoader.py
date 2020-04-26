
from torch.utils.data import Dataset





# Dataset 是抽象类，不能实例化，只能由其子类继承。记住就行了
class NyuHandDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=False, cache_size=15000):
        self.root = root
        self.normal_channel = normal_channel
        pass

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