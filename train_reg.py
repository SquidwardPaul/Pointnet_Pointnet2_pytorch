"""
Author: Jack Leung
Date: April 2020
Note：这是梁某基于分类的方法，魔改的手势关键点的回归
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.NyuHandDataLoader import NyuHandDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


#训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', default='pointnet2_reg_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_reg_msg', help='experiment root')
    parser.add_argument('--num_joint', default=36*3, type=int, help='number of joint in hand [default: 36*3]')

    parser.add_argument('--batch_size', type=int, default=6, help='batch size in training [default: 24]')
    parser.add_argument('--epoch',  default=10, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')

    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def test(model, loader):
    mean_dist = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        # 记录该 batch 的误差率
        dist = (pred - target) ** 2
        dist = dist.reshape(args.batch_size, 36, -1)
        dist = torch.sqrt(torch.sum(dist, 2))  # [Batch_size,36] 36个关键点的坐标，预测值与标签值的欧氏距离
        mean_dist.append(torch.mean(torch.mean(dist, 1)).item())  # 记录该 batch 的误差率

    # 计算该 epoch 的误差率
    test_instance_error = np.mean(mean_dist) # 该 epoch 下，总的正确率
    return test_instance_error


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR 建立保存路径'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('regression')
    experiment_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    '''LOG 建立日志文件'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING 加载数据'''
    log_string('Load dataset ...')
    DATA_PATH = './data/nyu_hand_dataset_v2/'
    # 这里是需要我改的 编写专用的数据集加载函数
    TRAIN_DATASET = NyuHandDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = NyuHandDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING 加载模型'''
    # 把依赖的模块函数 备份到日志路径里
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    MODEL = importlib.import_module(args.model) #导入模型所在的模块
    classifier = MODEL.get_model(args.num_joint, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda() # 计算损失函数

    # 尝试加载已有的训练模型
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    # 优化器 optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_error = 1000.0
    mean_dist = [] # 记录每次训练后，该batch的争取率


    '''TRANING 训练'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # 提取 mini_batch
            points, target = data
            # points 是当前 batch 的数据  target 是当前 batch 的标签
            points = points.data.numpy()    # points [B, Nsample, 坐标+法向量]
            points = provider.random_point_dropout(points, max_dropout_ratio=0.875) # points 做随机的 dropout
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])  # points 做随机比例的放缩
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])         # points 做随机幅度的位移
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            # 计算损失函数
            optimizer.zero_grad() # 梯度置零 把loss关于weight的导数变成0.
            classifier.train() # 使用PyTorch进行训练时,一定注意要把实例化的model指定train,表示启用 BatchNormalization 和 Dropout
            pred, trans_feat = classifier(points)   # pred 网络的输出， trans_feat 是输入的特征值，就是三个sa层后的输出
                                                    # pred [Batch_size, num_joint]
                                                    # trans_feat [Batch_size, SA层的输出大小, 1]
            loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()
            global_step += 1

            # 记录该 batch 的误差率
            dist = (pred - target) ** 2
            dist = dist.reshape(args.batch_size,36,-1)
            dist = torch.sqrt(torch.sum(dist, 2))  # [Batch_size,36] 36个关键点的坐标，预测值与标签值的欧氏距离
            mean_dist.append(torch.mean(torch.mean(dist,1)).item()) # 记录该 batch 的误差率

        # 计算该 epoch 的误差率
        train_instance_error = np.mean(mean_dist)
        log_string('Train Instance Error: %f' % train_instance_error)


        with torch.no_grad(): # 不需要计算梯度
            test_instance_error = test(classifier.eval(), testDataLoader)

            if (test_instance_error <= best_instance_error):
                best_instance_error = test_instance_error
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_error': best_instance_error,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            log_string('Test Instance Error: %f' % (test_instance_error))
            log_string('Best Instance Accuracy: %f'% (best_instance_error))

            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
