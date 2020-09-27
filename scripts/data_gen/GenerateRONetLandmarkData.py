# coding: utf-8
"""
读取LFW NET face 数据集
利用 P(+R)NET 生成正样本和负样本。
正样本标签为：1
负样本标签为：0
部分人脸样本标签为：2
landmark样本:-1
综合标签为： img_path xmin, ymin, xmax, ymax, landmark[10], label
例如： 1.jpg 1 -1 2 10 11
注意：保存的图片缩放到了N*N，bbox坐标也是相对于小图的，但是未做scale
"""
import sys
sys.path.append(sys.path[0] + "/../")
import os
import numpy as np
import random
import cv2
import lmdb
from util.Logger import Logger
import time
if not os.path.exists("./log"):
    os.mkdir("./log")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                      time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger
from MTCNN import *
from util.utility import boder_bbox

# 小于该人脸的就不要了
MIN_FACE_SIZE = 40
IOU_POS_THRES = 0.65
IOU_NEG_THRES = 0.3
IOU_PART_THRES = 0.4

## 关键点样本个数
landmark_samples = 6

net_type = "ONET"
if net_type == "RNET":
    OUT_IMAGE_SIZE = 24
    post_fix = 'r'
else:
    OUT_IMAGE_SIZE = 48
    post_fix = 'o'

root_dir = r'../../dataset/LFW_NET_FACE'
root_dir = os.path.expanduser(root_dir)

output_root_dir = r"../../dataset/train_faces_{}".format(post_fix)

if not os.path.exists(output_root_dir):
    os.mkdir(output_root_dir)

output_landmark_dir = os.path.join(output_root_dir, "landmark")
if not os.path.exists(output_landmark_dir):
    os.mkdir(output_landmark_dir)


def GenerateData(mt):
    LMDB_MAP_SIZE = 1099511627776
    env_landmark_image = lmdb.open(os.path.join(output_landmark_dir, "image_landmark"), map_size=LMDB_MAP_SIZE)
    env_landmark_label = lmdb.open(os.path.join(output_landmark_dir, "label_landmark"), map_size=LMDB_MAP_SIZE)
    global_idx_landmark = 0
    txn_landmark_image = env_landmark_image.begin(write=True)
    txn_landmark_label = env_landmark_label.begin(write=True)
    anno_file = os.path.join(root_dir, "trainImageList.txt")
    with open(anno_file, "r") as f:
        inner_landmark_idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            line_split = line.split()
            filename_split = line_split[0].split("\\")
            filename = os.path.join(filename_split[0],filename_split[1])
            img = cv2.imread(os.path.join(root_dir,filename))
            if img is None:
                log.warning("error to load image {}", filename)
                continue
            # 读取真值 bbox
            H, W, C = img.shape
            x = int(line_split[1])
            x1 = int(line_split[2])
            y = int(line_split[3])
            y1 = int(line_split[4])
            w = x1 - x
            h = y1 - y
            gt_bbox = np.array([x, y, w, h])
            bbox = mt.detect(img)
            if bbox is None:
                continue
            bbox = bbox.astype(np.int32)
            bbox = bbox[:, 0:4]
            for i in bbox:
                iou = IOU(i, gt_bbox)
                if iou < IOU_PART_THRES:
                    continue
                r = square_bbox(i)
                r_p = boder_bbox(r, 10)

                size = r[2]
                size_p = r_p[2]
                crop = np.zeros((size, size, 3), dtype=np.uint8)
                temp_crop = np.zeros((size_p, size_p, 3), dtype=np.uint8)
                sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(r_p, W, H)

                if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size_p or dy1 > size_p:
                    log.warning("img shape is: {},{}".format(img.shape[0], img.shape[1]))
                    continue
                rotation = [-30, -15, 0, 15, 30] 
                temp_crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
                for theta in rotation:
                    (h_, w_) = temp_crop.shape[:2] 
                    center = (w_ // 2, h_ // 2)
                    M = cv2.getRotationMatrix2D(center, -theta, 1.0) #12
                    croped = cv2.warpAffine(temp_crop, M, (w_, h_)) #13
                    crop = croped[10:-10,10:-10]
                    center_x = r[0] + r[2]/2
                    center_y = r[1] + r[3]/2
                    landmarks = []
                    for i in range(5):
                        x_ = int(float(line_split[5+i*2])) 
                        y_ = int(float(line_split[5+i*2+1])) 
                        r_x = (x_ - center_x) * np.cos(theta/180*3.14159) - (y_ - center_y)*np.sin(theta/180*3.14159) + center_x
                        r_y = (x_ - center_x) * np.sin(theta/180*3.14159) + (y_ - center_y)*np.cos(theta/180*3.14159) + center_y
                        landmarks.append(r_x)
                        landmarks.append(r_y)

                    out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))

                    # ground truth 坐标变换
                    ## 变换到crop为原点的坐标
                    ground_truth = gt_bbox
                    ox = ground_truth[0] - r[0]
                    oy = ground_truth[1] - r[1]
                    ow = ground_truth[2]
                    oh = ground_truth[3]

                    ## 变换到out
                    scalor = float(size) / float(OUT_IMAGE_SIZE)
                    ox = ox / scalor
                    oy = oy / scalor
                    ow = ow / scalor
                    oh = oh / scalor

                    # # 这里保存成左上角点和右下角点
                    xmin = ox
                    ymin = oy
                    xmax = xmin + ow
                    ymax = ymin + oh

                    #关键点
                    #landmarks = []
                    for i in range(5):
                        landmarks[i*2] = int(landmarks[i*2]) - r[0]
                        landmarks[i*2+1] = int(landmarks[i*2+1]) - r[1]
                        landmarks[i*2] = landmarks[i*2] / scalor
                        landmarks[i*2+1] = landmarks[i*2+1] / scalor

                    if iou > IOU_POS_THRES:
                        label_list = [xmin, ymin, xmax, ymax] + landmarks + [-1]
                        label = np.array(label_list, dtype=np.float32)
                        txn_landmark_image.put("{}".format(global_idx_landmark).encode("ascii"), out.tostring())
                        txn_landmark_label.put("{}".format(global_idx_landmark).encode("ascii"), label.tostring())
                        global_idx_landmark += 1
                        inner_landmark_idx += 1
                    log.info("landmark num: {}".format(global_idx_landmark))
                    if inner_landmark_idx > 1000:
                        txn_landmark_image.commit()
                        txn_landmark_label.commit()
                        txn_landmark_image = env_landmark_image.begin(write=True)
                        txn_landmark_label = env_landmark_label.begin(write=True)
                        inner_landmark_idx = 0
                        log.info("now commit landmark lmdb")
    
    anno_file = os.path.join(root_dir, "testImageList.txt")
    with open(anno_file, "r") as f:
        inner_landmark_idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            line_split = line.split()
            filename_split = line_split[0].split("\\")
            filename = os.path.join(filename_split[0],filename_split[1])
            img = cv2.imread(os.path.join(root_dir,filename))
            if img is None:
                log.warning("error to load image {}", filename)
                continue
            # 读取真值 bbox
            H, W, C = img.shape
            x = int(line_split[1])
            x1 = int(line_split[2])
            y = int(line_split[3])
            y1 = int(line_split[4])
            w = x1 - x
            h = y1 - y
            gt_bbox = np.array([x, y, w, h])
            bbox = mt.detect(img)
            if bbox is None:
                continue
            bbox = bbox.astype(np.int32)
            bbox = bbox[:, 0:4]
            for i in bbox:
                iou = IOU(i, gt_bbox)
                if iou < IOU_PART_THRES:
                    continue
                r = square_bbox(i)
                r_p = boder_bbox(r, 10)

                size = r[2]
                size_p = r_p[2]
                crop = np.zeros((size, size, 3), dtype=np.uint8)
                temp_crop = np.zeros((size_p, size_p, 3), dtype=np.uint8)
                sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(r_p, W, H)

                if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size_p or dy1 > size_p:
                    log.warning("img shape is: {},{}".format(img.shape[0], img.shape[1]))
                    continue
                rotation = [-30, -15, 0, 15, 30] 
                temp_crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
                for theta in rotation:
                    (h_, w_) = temp_crop.shape[:2] 
                    center = (w_ // 2, h_ // 2)
                    M = cv2.getRotationMatrix2D(center, -theta, 1.0) #12
                    croped = cv2.warpAffine(temp_crop, M, (w_, h_)) #13
                    crop = croped[10:-10,10:-10]
                    center_x = r[0] + r[2]/2
                    center_y = r[1] + r[3]/2
                    landmarks = []
                    for i in range(5):
                        x_ = int(float(line_split[5+i*2])) 
                        y_ = int(float(line_split[5+i*2+1])) 
                        r_x = (x_ - center_x) * np.cos(theta/180*3.14159) - (y_ - center_y)*np.sin(theta/180*3.14159) + center_x
                        r_y = (x_ - center_x) * np.sin(theta/180*3.14159) + (y_ - center_y)*np.cos(theta/180*3.14159) + center_y
                        landmarks.append(r_x)
                        landmarks.append(r_y)

                    out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))

                    # ground truth 坐标变换
                    ## 变换到crop为原点的坐标
                    ground_truth = gt_bbox
                    ox = ground_truth[0] - r[0]
                    oy = ground_truth[1] - r[1]
                    ow = ground_truth[2]
                    oh = ground_truth[3]

                    ## 变换到out
                    scalor = float(size) / float(OUT_IMAGE_SIZE)
                    ox = ox / scalor
                    oy = oy / scalor
                    ow = ow / scalor
                    oh = oh / scalor

                    # # 这里保存成左上角点和右下角点
                    xmin = ox
                    ymin = oy
                    xmax = xmin + ow
                    ymax = ymin + oh

                    #关键点
                    #landmarks = []
                    for i in range(5):
                        landmarks[i*2] = int(landmarks[i*2]) - r[0]
                        landmarks[i*2+1] = int(landmarks[i*2+1]) - r[1]
                        landmarks[i*2] = landmarks[i*2] / scalor
                        landmarks[i*2+1] = landmarks[i*2+1] / scalor

                    if iou > IOU_POS_THRES:
                        label_list = [xmin, ymin, xmax, ymax] + landmarks + [-1]
                        label = np.array(label_list, dtype=np.float32)
                        txn_landmark_image.put("{}".format(global_idx_landmark).encode("ascii"), out.tostring())
                        txn_landmark_label.put("{}".format(global_idx_landmark).encode("ascii"), label.tostring())
                        global_idx_landmark += 1
                        inner_landmark_idx += 1
                    log.info("landmark num: {}".format(global_idx_landmark))
                    if inner_landmark_idx > 1000:
                        txn_landmark_image.commit()
                        txn_landmark_label.commit()
                        txn_landmark_image = env_landmark_image.begin(write=True)
                        txn_landmark_label = env_landmark_label.begin(write=True)
                        inner_landmark_idx = 0
                        log.info("now commit landmark lmdb")

    log.info("process done!")
    txn_landmark_image.commit()
    txn_landmark_label.commit()

if __name__ == "__main__":
    USE_CUDA = True
    GPU_ID = [0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")
    # pnet
    pnet_weight_path = "../models/pnet_20200917_final.pkl"
    pnet = PNet(test=True)
    LoadWeights(pnet_weight_path, pnet)
    pnet.to(device)

    # rnet
    rnet = None
    if net_type == "ONET":
        rnet_weight_path = "../models/rnet_20200917_final.pkl"
        rnet = RNet(test=True)
        LoadWeights(rnet_weight_path, rnet)
        rnet.to(device)

    mt = MTCNN(detectors=[pnet, rnet, None], min_face_size=24, threshold=[0.5, 0.5, 0.5], device=device)
    GenerateData(mt)
    log.info("over...")