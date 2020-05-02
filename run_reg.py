"""
Author: Jack Leung
Date: May 2020
调用训练好的模型
"""

import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import matplotlib
from pathlib import Path
import sys
import importlib
import cv2
from openni import openni2
from openni import _openni2 as c_api
from displayPoint import displayPoint1,displayPoint2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('')
    # 模型 参数
    parser.add_argument('--num_joint', default=6, type=int, help='number of joint in hand [default: 36*3]')
    parser.add_argument('--model', default='pointnet2_reg_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')

    # 摄像机参数
    parser.add_argument('--width', type=int, default=640, help='resolutionX')
    parser.add_argument('--height', type=int, default=400, help='resolutionY')
    parser.add_argument('--fps', type=int, default=30, help='frame per second')
    parser.add_argument('--mirroring', default=True, help='mirroring [default: False]')
    parser.add_argument('--compression', default=True, help='compress or not, when saving the video [default: True]')

    return parser.parse_args()


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / scale
    return pc, scale, centroid


def getOrbbec():
    # 记载 openni
    try:
        if sys.platform == "win32":
            libpath = "lib/Windows"
        else:
            libpath = "lib/Linux"
        print("library path is: ", os.path.join(os.path.dirname(__file__), libpath))
        openni2.initialize(os.path.join(os.path.dirname(__file__), libpath))
        print("OpenNI2 initialized \n")
    except Exception as ex:
        print("ERROR OpenNI2 not initialized", ex, " check library path..\n")
        return

    # 加载 orbbec 相机
    try:
        device = openni2.Device.open_any()
        return device
    except Exception as ex:
        print("ERROR Unable to open the device: ", ex, " device disconnected? \n")
        return

def depth2uvd(depth_array):
    U = np.tile(np.linspace(1, args.width, args.width), (args.height, 1)).astype(np.float32)
    V = np.tile(np.linspace(1, args.height, args.height), (args.width, 1)).astype(np.float32).transpose(1, 0)
    cloud_z = depth_array
    cloud_x = ((U - 309.9648) * cloud_z) / 515.8994 + (244.1680 - V) * 1.3982e-06 * cloud_z
    cloud_y = (V - 244.1680) * cloud_z / 516.2843
    # 下采样
    cloud_x = cloud_x[0::3, 0::3]
    cloud_y = cloud_y[0::3, 0::3]
    cloud_z = cloud_z[0::3, 0::3]
    # 变换成标准形式 cloud_point[Num, 3],并删除无效点
    cloud_x = np.reshape(cloud_x, (-1, 1))
    cloud_y = np.reshape(cloud_y, (-1, 1))
    cloud_z = np.reshape(cloud_z, (-1, 1))
    cloud_point = np.hstack((cloud_x, cloud_y, cloud_z))
    index = np.where(cloud_point[:, 2] == 0)[0]
    cloud_point = np.delete(cloud_point, index, axis=0)
    index = np.where(cloud_point[:, 2] > 2000)[0]
    cloud_point = np.delete(cloud_point, index, axis=0)

    return cloud_point

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''MODEL LOADING 加载模型'''
    MODEL = importlib.import_module(args.model)  # 导入模型所在的模块
    classifier = MODEL.get_model(args.num_joint * 3, normal_channel=args.normal).cuda()

    experiment_dir = Path('./log/regression/pointnet2_reg_msg')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()


    try:
        ''' 加载摄像机 '''
        device = getOrbbec()
        # 创建深度流
        depth_stream = device.create_depth_stream()
        depth_stream.set_mirroring_enabled(args.mirroring)
        depth_stream.set_video_mode(c_api.OniVideoMode(resolutionX=args.width, resolutionY=args.height, fps=args.fps,
                                                       pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM))
        # 获取uvc
        cap = cv2.VideoCapture(0)

        # 设置 镜像 帧同步
        device.set_image_registration_mode(True)
        device.set_depth_color_sync_enabled(True)
        depth_stream.start()

        while True:
            # 读取帧
            frame_depth = depth_stream.read_frame()
            frame_depth_data = frame_depth.get_buffer_as_uint16()
            # 读取帧的深度信息 depth_array 也是可以用在后端处理的 numpy格式的
            depth_array = np.ndarray((frame_depth.height, frame_depth.width), dtype=np.uint16, buffer=frame_depth_data)
            # 变换格式用于 opencv 显示
            depth_uint8 = 1 - 250 / (depth_array)
            depth_uint8[depth_uint8 > 1] = 1
            depth_uint8[depth_uint8 < 0] = 0
            cv2.imshow('depth', depth_uint8)

            # 读取 彩色图
            _, color_array = cap.read()
            cv2.imshow('color', color_array)

            # 对彩色图 color_array 做处理

            # 对深度图 depth_array 做处理

            # 键盘监听
            if cv2.waitKey(1) == ord('q'):
                # 关闭窗口 和 相机
                depth_stream.stop()
                cap.release()
                cv2.destroyAllWindows()
                break

        # 检测设备是否关闭（没什么用）
        try:
            openni2.unload()
            print("Device unloaded \n")
        except Exception as ex:
            print("Device not unloaded: ", ex, "\n")


    except:
        # 读取 depth
        depth_array = matplotlib.image.imread('test_depth_1.tif').astype(np.float32)
        # depth to UVD
        cloud_point = depth2uvd(depth_array)
        # 将点云归一化
        cloud_point_normal, scale, centroid = pc_normalize(cloud_point)
        cloud_point_normal = np.reshape(cloud_point_normal,(1,-1,3))
        cloud_point_normal = cloud_point_normal.transpose(0, 2, 1)
        # 对归一化的点云做预测
        cloud_point_normal = torch.from_numpy(cloud_point_normal).cuda()
        pred, _ = classifier(cloud_point_normal)
        # 对预测结果做还原
        pred_reduction = pred.cpu().data.numpy()
        pred_reduction = pred_reduction * np.tile(scale, (args.num_joint * 3, 1)).transpose(1, 0)
        pred_reduction = pred_reduction + np.tile(centroid, (1, args.num_joint))
        pred_reduction = np.reshape(pred_reduction,(-1,3))

        displayPoint2(cloud_point, pred_reduction, 'bear')






    # with torch.no_grad():
    #     classifier.eval()
    #     instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)




if __name__ == '__main__':
    args = parse_args()
    main(args)
