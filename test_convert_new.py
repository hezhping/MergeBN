#coding=utf-8

import os.path as osp
import sys
import copy
import os
import cv2
import numpy as np

CAFFE_ROOT = '/root/RefineDet/'
if osp.join(CAFFE_ROOT,'python') not in sys.path:
        sys.path.insert(0,osp.join(CAFFE_ROOT,'python'))

import caffe

GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
net = caffe.Net('./eid_naive_train_mg.prototxt', './res50_max_tzwd_redstar_wcc_iter_80000.caffemodel', caffe.TEST)
#result_net = caffe.Net('./models/result.prototxt', './models/result.caffemodel', caffe.TEST)

net.blobs['data'].reshape(1, 3, 384, 128)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))
image = caffe.io.load_image('/ssd/yfchen/data/fisheye_benchmark/CTF_xhm_FID_fisheye_Body_App_benchmark/fisheye_xhm_1031_3231x/00323170/ch02003_20181031182411_pano_00323170_00010416.jpg')
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image
res = net.forward()
feat_1 = res['pool_res5c'].flatten()

net = caffe.Net('./merged_eid_naive_train_mg.prototxt', './merged_res50_max_tzwd_redstar_wcc_iter_80000.caffemodel', caffe.TEST)
#result_net = caffe.Net('./models/result.prototxt', './models/result.caffemodel', caffe.TEST)

net.blobs['data'].reshape(1, 3, 384, 128)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))
image = caffe.io.load_image('/ssd/yfchen/data/fisheye_benchmark/CTF_xhm_FID_fisheye_Body_App_benchmark/fisheye_xhm_1031_3231x/00323170/ch02003_20181031182411_pano_00323170_00010416.jpg')
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image
res = net.forward()
feat_2 = res['pool_res5c'].flatten()

print feat_1
print feat_2
print sum(np.absolute(feat_1-feat_2))
