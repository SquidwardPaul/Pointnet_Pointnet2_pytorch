import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 三维离散点图显示点云
def displayPoint(data, title):
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 点数量太多不予显示
    while len(data[0]) > 20000:
        print("点太多了！")
        exit()
    # 散点图参数设置
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r', marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


