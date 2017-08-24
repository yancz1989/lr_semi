# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-05-08 11:23:55
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-17 07:40:53

import cv2
import numpy as np
import scipy as sp
import argparse
import logging
import json

import mxnet as mx
import fit
from nuclear import NuclearProp, Nuclear

gpu = 2
workspace = 256
mu = 29.5565373846
sigma = 63.7321981505

DATA_ROOT = './data/simple/'
INPUT_ROOT = DATA_ROOT + 'input/'
SEG_ROOT = DATA_ROOT + 'seg/'

def segnet(workspace = 512):
  network = {}
  network['data'] = mx.symbol.Variable('data')
  # conv 1
  network['conv1_1'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['data'],
    kernel = (5, 5), num_filter = 32, pad = (2, 2), workspace = workspace), act_type = 'relu', name = 'conv1_1')
  network['conv1_2'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['conv1_1'],
    kernel = (3, 3), num_filter = 32, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'conv1_2')
  network['pool1'] = mx.symbol.Pooling(data = network['conv1_2'],
    pool_type = 'max', kernel = (2, 2), stride = (2, 2), name = 'pool1')

  # # conv 2
  network['conv2_1'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['pool1'],
    kernel = (3, 3), num_filter = 64, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'conv2_1')
  network['conv2_2'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['conv2_1'],
    kernel = (3, 3), num_filter = 64, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'conv2_2')
  network['pool2'] = mx.symbol.Pooling(data = network['conv2_2'],
    pool_type = "max", kernel = (2, 2), stride = (2, 2), name = 'pool2')

  # conv 3
  network['conv3'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['pool2'],
    kernel = (3, 3), num_filter = 128, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'conv3')
  network['pool3'] = mx.symbol.Pooling(data = network['conv3'],
    pool_type = 'max', kernel = (2, 2), stride = (2, 2), name = 'pool3')
  encoder = network['pool3']

  # deconv1
  network['upspl3'] = mx.symbol.Activation(mx.symbol.UpSampling(
      data = encoder,
      scale = 2,
      num_filter = 128,
      sample_type = 'bilinear',
      num_args = 2,
      workspace = workspace),
    act_type = 'relu', name = 'upspl3')
  network['deconv3'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['upspl3'], 
    kernel = (3, 3), num_filter = 128, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'deconv3')

  # deconv 2
  network['upspl2'] = mx.symbol.Activation(mx.symbol.UpSampling(
      data = network['deconv3'],
      scale = 2,
      num_filter = 128,
      sample_type = 'bilinear',
      num_args = 2,
      workspace = workspace),
    act_type = 'relu', name = 'upspl2')
  network['deconv2_2'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['upspl2'], 
    kernel = (3, 3), num_filter = 64, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'deconv2_2')
  network['deconv2_1'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['deconv2_2'], 
    kernel = (3, 3), num_filter = 16, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'deconv2_1')
  # network['output'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['deconv2_1'], 
  #   kernel = (3, 3), num_filter = 1, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'output')

  # deconv 1
  # network['upspl1'] = mx.symbol.Activation(mx.symbol.UpSampling(
  #     data = network['deconv2_1'],
  #     scale = 2,
  #     num_filter = 128,
  #     sample_type = 'bilinear',
  #     num_args = 2,
  #     workspace = workspace),
  #   act_type = 'relu', name = 'upspl1')
  # network['deconv1_2'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['upspl1'], 
  #   kernel = (3, 3), num_filter = 32, pad = (1, 1), workspace = workspace), act_type = 'relu', name = 'deconv1_2')
  # network['deconv1_1'] = mx.symbol.Activation(mx.symbol.Convolution(data = network['deconv1_2'], 
  #   kernel = (3, 3), num_filter = 1, pad = (2, 2), workspace = workspace), act_type = 'relu', name = 'deconv1_1')
  # network['region'] = mx.symbol.slice(data = network['deconv1_1'], begin = (None, 0, 0), end = (None, 396, 224))



  # network['softmax'] = mx.symbol.SoftmaxOutput(data = network['deconv2_1'],
  #   multi_output = True, use_ignore = True, ignore_label = 2, name = 'softmax')


  # network['softmax_'] = mx.symbol.Softmax(data = network['deconv2_1'], axis = 1)
  network['Nuclear'] = mx.symbol.Custom(data = network['deconv2_1'], name = 'Nuclear', op_type='Nuclear')

  decoder = network['Nuclear']

  return (network, encoder, decoder)


def to4d(dat):
  return dat.reshape(dat.shape[0], 1, dat.shape[1], dat.shape[2])


def data_iter(args, kv):
  with open('./data/meta/part_train.json', 'r') as f:
    tflist = json.load(f)
  with open('./data/meta/part_val.json', 'r') as f:
    vflist = json.load(f)
  ltrain, lval, ltlabel, lvlabel = [], [], [], []
  for f in tflist[:32]:
    ltrain.append(cv2.imread(INPUT_ROOT + f, 0))
    ltlabel.append(cv2.resize(cv2.imread(SEG_ROOT + f, 0), fx = 0.5, fy = 0.5, dsize = (396, 224)))
    # ltlabel.append(cv2.imread(SEG_ROOT + f, 0))
  ltrain = ltrain[-16:]
  ltlabel = ltlabel[-16:]
  for f in vflist[:16]:
    lval.append(cv2.imread(INPUT_ROOT + f, 0))
    lvlabel.append(cv2.resize(cv2.imread(SEG_ROOT + f, 0), fx = 0.5, fy = 0.5, dsize = (396, 224)))
    # lvlabel.append(cv2.imread(SEG_ROOT + f, 0))


  np_train = (to4d(np.array(ltrain, dtype = np.float32)) - mu) / sigma
  np_ltrain = to4d(np.array(ltlabel, dtype = np.int32))
  np_val = (to4d(np.array(lval, dtype = np.float32)) - mu) / sigma
  np_lval = to4d(np.array(lvlabel, dtype = np.int32))

  train = mx.io.NDArrayIter(np_train, np_ltrain, args.batch_size, shuffle=True, label_name = 'Nuclear_label')
  val = mx.io.NDArrayIter(np_val, np_lval, args.batch_size, label_name = 'Nuclear_label')
  return (train, val, len(np_train), len(np_val))


def main():
  parser = argparse.ArgumentParser(description='train unet',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  fit.add_fit_args(parser)
  seed = 2012310818
  np.random.seed(seed)
  parser.set_defaults(
      batch_size=16,
      num_epochs=40,
      lr=0.0001,
      optimizer = 'adam'
      # lr_step_epochs='10',
      # lr_factor = 0.8
  )
  args = parser.parse_args()
  kv = mx.kvstore.create(args.kv_store)

  unet = segnet(workspace = workspace)
  train, val, lt, lv = data_iter(args, kv)
  fit.fit(args, unet[2], data_iter)


if __name__ == '__main__':
  main()
