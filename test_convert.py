#coding=utf-8

import os.path as osp
import sys
import copy
import os
import cv2
import numpy as np
import caffe
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='convert prototxt to prototxt without batch normalization')
    parser.add_argument('--model', 
                        default="", 
                        type=str)
    parser.add_argument('--weights', 
                        default="", 
                        type=str)
    parser.add_argument('--merged-model',
                        default="",
                        type=str)
    parser.add_argument('--merged-weights',
                        default="",
                        type=str)
    parser.add_argument('--feat_blob',
                        default="reid",
                        type=str)
    parser.add_argument('--input_h',
                        default=384,
                        type=int)
    parser.add_argument('--input_w',
                        default=128,
                        type=int)
    parser.add_argument('--GPU_ID',
                        default=0,
                        type=int)
    args = parser.parse_args()
    return args

def compare(model1, weights1, model2, weights2, feat_blob, net_h, net_w, GPU_ID=0):
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    net = caffe.Net(model1, weights1, caffe.TEST)

    net.blobs['data'].reshape(1, 3, net_h, net_w)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))
    image = caffe.io.load_image('./test.jpg')
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = np.reshape(transformed_image, (1, 3, net_h, net_w))
    res = net.forward()
    feat_1 = np.array(res[feat_blob].flatten().copy(), dtype=np.float32)

    net = caffe.Net(model2, weights2, caffe.TEST)
    net.blobs['data'].reshape(1, 3, net_h, net_w)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))
    image = caffe.io.load_image('./test.jpg')
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = np.reshape(transformed_image, (1, 3, net_h, net_w))
    res = net.forward()

    feat_2 = np.array(res[feat_blob].flatten().copy(), dtype=np.float32)
    diff = sum(np.absolute(feat_1-feat_2))
    print('diff:%.4f' % diff)
    mean_diff = diff / len(feat_1)
    print('mean_diff:%.4f' % mean_diff)
    relative_diff = diff / np.max(np.absolute(feat_1))
    print('relative_diff:%.4f' % relative_diff)

    feat_1 /= np.linalg.norm(feat_1)
    feat_2 /= np.linalg.norm(feat_2)
    sim = np.dot(feat_1, feat_2) 
    print('sim:%.4f' % sim)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    compare(args.model, args.weights, args.merged_model, args.merged_weights, args.feat_blob, args.input_h, args.input_w, args.GPU_ID)
