# coding: utf-8
"""
读取lfw net face 数据集
生成正样本和负样本。
正样本标签为：1
负样本标签为：0
部分人脸样本标签为：2
landmark样本:-1
综合标签为： img_path xmin ymin xmax ymax landmark[10] label

注意：保存的图片缩放到了12*12，bbox的坐标也是相对于12*12的
"""
import sys
sys.path.append(sys.path[0] + "/../")
import os
import numpy as np
import random
import cv2
from pylab import plt
from util.utility import *
from util.Logger import Logger
import time
import lmdb
if not os.path.exists("./log"):
    os.mkdir("./log")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                      time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger
# 小于该人脸的就不要了
# 太小的话，会有太多的误检
MIN_FACE_SIZE = 40
IOU_POS_THRES = 0.65
IOU_NEG_THRES = 0.3
IOU_PART_THRES = 0.4

## 关键点样本个数
landmark_samples = 20

OUT_IMAGE_SIZE = 12

root_dir = r'../../dataset/LFW_NET_FACE'
root_dir = os.path.expanduser(root_dir)
output_root_dir = r"../../dataset/train_faces_p"
if not os.path.exists(output_root_dir):
    os.mkdir(output_root_dir)
output_landmark_dir = os.path.join(output_root_dir, "landmark")
if not os.path.exists(output_landmark_dir):
    os.mkdir(output_landmark_dir)

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
        box = np.array([x, y, w, h])
        for i in range(landmark_samples):
            size = random.randrange(int(np.min((w,h)) * 0.8), int(np.ceil(1.25 * np.max((w,h)))))
            dx = random.randrange(int(-w * 0.2), int(w * 0.2))
            dy = random.randrange(int(-h * 0.2), int(h * 0.2))
            nx = np.max((x + w / 2 + dx - size / 2), 0)
            ny = np.max((y + h / 2 + dy - size / 2), 0)
            nx = int(nx)
            ny = int(ny)
            if nx < 0:
                nx = 0
            if ny < 0:
                ny = 0
            if nx + size > W or ny + size > H:
                continue
            if size < OUT_IMAGE_SIZE / 2:
                continue

            #iou
            crop_box = np.array([nx, ny, size, size])
            iou = IOU(box, crop_box)
            # log.info("{} {} {} {}".format(nx, ny, size, size))
            crop = img[ny: ny+size, nx:nx+size]
            out = cv2.resize(crop, (OUT_IMAGE_SIZE, OUT_IMAGE_SIZE))
            scalor = float(size) / float(OUT_IMAGE_SIZE)
            # ground truth 坐标变换
            ## 变换到crop
            ox = x - nx
            oy = y - ny
            ow = w
            oh = h
            # ## 变换到out
            ox = ox / scalor
            oy = oy / scalor
            ow = ow / scalor
            oh = oh / scalor
            #关键点
            landmarks = []
            for i in range(5):
                lx = int(float(line_split[5+i*2])) - nx
                ly = int(float(line_split[5+i*2+1])) - ny
                lx = lx / scalor
                ly = ly / scalor
                landmarks.append(lx)
                landmarks.append(ly)
            #
            # # 这里保存成左上角点和右下角点
            xmin = ox
            ymin = oy
            xmax = xmin + ow
            ymax = ymin + oh
            if iou > IOU_POS_THRES:
                #### 正样本
                #path_ = "/home/chengliu/MTCNN/mtcnn-pytorch/dataset/out_img/" + str(global_idx_landmark) + ".png"
                #cv2.imwrite(path_,out)
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