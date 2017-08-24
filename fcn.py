# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-05-08 11:23:35
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-06-07 16:46:32
# pylint: skip-file
import sys
import os
import argparse
import mxnet as mx
import numpy as np
import logging
# import symbol
# import init_fcnxs
from dataset import FileIter
from solver import Solver

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(2)


def get_lenet(num_classes=10, add_stn=False, **kwargs):
  data = mx.symbol.Variable('data')
  if(add_stn):
    data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape=(28, 28),
                                     transform_type="affine", sampler_type="bilinear")
  # first conv
  conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
  tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
  pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                            kernel=(2, 2), stride=(2, 2))
  # second conv
  conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
  tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
  pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                            kernel=(2, 2), stride=(2, 2))
  # first fullc
  flatten = mx.symbol.Flatten(data=pool2)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
  # loss
  lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
  return lenet


def filter_map(kernel=1, stride=1, pad=0):
  return (stride, (kernel - stride) / 2 - pad)


def compose_fp(fp_first, fp_second):
  return (fp_first[0] * fp_second[0], fp_first[0] * fp_second[1] + fp_first[1])


def compose_fp_list(fp_list):
  fp_out = (1.0, 0.0)
  for fp in fp_list:
    fp_out = compose_fp(fp_out, fp)
  return fp_out


def inv_fp(fp_in):
  return (1.0 / fp_in[0], -1.0 * fp_in[1] / fp_in[0])


def offset():
  conv1_1_fp = filter_map(kernel=3, pad=100)
  conv1_2_fp = conv2_1_fp = conv2_2_fp = conv3_1_fp = conv3_2_fp = conv3_3_fp \
      = conv4_1_fp = conv4_2_fp = conv4_3_fp = conv5_1_fp = conv5_2_fp \
      = conv5_3_fp = filter_map(kernel=3, pad=1)
  pool1_fp = pool2_fp = pool3_fp = pool4_fp = pool5_fp = filter_map(
      kernel=2, stride=2)
  fc6_fp = filter_map(kernel=7)
  fc7_fp = score_fp = score_pool4_fp = score_pool3_fp = filter_map()
  # for fcn-32s
  fcn32s_upscore_fp = inv_fp(filter_map(kernel=64, stride=32))
  fcn32s_upscore_list = [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp,
                         pool2_fp, conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                         conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp, conv5_1_fp,
                         conv5_2_fp, conv5_3_fp, pool5_fp, fc6_fp, fc7_fp, score_fp,
                         fcn32s_upscore_fp]
  crop = {}
  crop["fcn32s_upscore"] = (-int(round(compose_fp_list(fcn32s_upscore_list)[1])),
                            -int(round(compose_fp_list(fcn32s_upscore_list)[1])))
  # for fcn-16s
  score2_fp = inv_fp(filter_map(kernel=4, stride=2))
  fcn16s_upscore_fp = inv_fp(filter_map(kernel=32, stride=16))
  score_pool4c_fp_list = [inv_fp(score2_fp), inv_fp(score_fp), inv_fp(fc7_fp), inv_fp(fc6_fp),
                          inv_fp(pool5_fp), inv_fp(conv5_3_fp), inv_fp(conv5_2_fp),
                          inv_fp(conv5_1_fp), score_pool4_fp]
  crop["score_pool4c"] = (-int(round(compose_fp_list(score_pool4c_fp_list)[1])),
                          -int(round(compose_fp_list(score_pool4c_fp_list)[1])))
  fcn16s_upscore_list = [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp,
                         pool2_fp, conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                         conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp, score_pool4_fp,
                         inv_fp((1, -crop["score_pool4c"][0])), fcn16s_upscore_fp]
  crop["fcn16s_upscore"] = (-int(round(compose_fp_list(fcn16s_upscore_list)[1])),
                            -int(round(compose_fp_list(fcn16s_upscore_list)[1])))
  # for fcn-8s
  score4_fp = inv_fp(filter_map(kernel=4, stride=2))
  fcn8s_upscore_fp = inv_fp(filter_map(kernel=16, stride=8))
  score_pool3c_fp_list = [inv_fp(score4_fp), (1, -crop["score_pool4c"][0]), inv_fp(score_pool4_fp),
                          inv_fp(pool4_fp), inv_fp(conv4_3_fp), inv_fp(conv4_2_fp),
                          inv_fp(conv4_1_fp), score_pool3_fp, score_pool3_fp]
  crop["score_pool3c"] = (-int(round(compose_fp_list(score_pool3c_fp_list)[1])),
                          -int(round(compose_fp_list(score_pool3c_fp_list)[1])))
  fcn8s_upscore_list = [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp, pool2_fp,
                        conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp, score_pool3_fp,
                        inv_fp((1, -crop["score_pool3c"][0])), fcn8s_upscore_fp]
  crop["fcn8s_upscore"] = (-int(round(compose_fp_list(fcn8s_upscore_list)[1])),
                           -int(round(compose_fp_list(fcn8s_upscore_list)[1])))
  return crop


def vgg16_pool3(input, workspace_default=1024):
  # group 1
  conv1_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(100, 100), num_filter=64,
                                  workspace=workspace_default, name="conv1_1")
  relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
  conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                                  workspace=workspace_default, name="conv1_2")
  relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
  pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max",
                            kernel=(2, 2), stride=(2, 2), name="pool1")
  # group 2
  conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                                  workspace=workspace_default, name="conv2_1")
  relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
  conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                                  workspace=workspace_default, name="conv2_2")
  relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
  pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max",
                            kernel=(2, 2), stride=(2, 2), name="pool2")
  # group 3
  conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                  workspace=workspace_default, name="conv3_1")
  relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
  conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                  workspace=workspace_default, name="conv3_2")
  relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
  conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                                  workspace=workspace_default, name="conv3_3")
  relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
  pool3 = mx.symbol.Pooling(data=relu3_3, pool_type="max",
                            kernel=(2, 2), stride=(2, 2), name="pool3")
  return pool3


def vgg16_pool4(input, workspace_default=1024):
  # group 4
  conv4_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                  workspace=workspace_default, name="conv4_1")
  relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
  conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                  workspace=workspace_default, name="conv4_2")
  relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
  conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                  workspace=workspace_default, name="conv4_3")
  relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
  pool4 = mx.symbol.Pooling(data=relu4_3, pool_type="max",
                            kernel=(2, 2), stride=(2, 2), name="pool4")
  return pool4


def vgg16_score(input, numclass, workspace_default=1024):
  # group 5
  conv5_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                  workspace=workspace_default, name="conv5_1")
  relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
  conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                  workspace=workspace_default, name="conv5_2")
  relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
  conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                                  workspace=workspace_default, name="conv5_3")
  relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
  pool5 = mx.symbol.Pooling(data=relu5_3, pool_type="max",
                            kernel=(2, 2), stride=(2, 2), name="pool5")
  # group 6
  fc6 = mx.symbol.Convolution(data=pool5, kernel=(7, 7), num_filter=4096,
                              workspace=workspace_default, name="fc6")
  relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
  drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
  # group 7
  fc7 = mx.symbol.Convolution(data=drop6, kernel=(1, 1), num_filter=4096,
                              workspace=workspace_default, name="fc7")
  relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
  drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
  # group 8
  score = mx.symbol.Convolution(data=drop7, kernel=(1, 1), num_filter=numclass,
                                workspace=workspace_default, name="score")
  return score


def fcnxs_score(input, crop, offset, kernel=(64, 64), stride=(32, 32), numclass=21, workspace_default=1024):
  # score out
  bigscore = mx.symbol.Deconvolution(data=input, kernel=kernel, stride=stride, adj=(stride[0] - 1, stride[1] - 1),
                                     num_filter=numclass, workspace=workspace_default, name="bigscore")
  upscore = mx.symbol.Crop(*[bigscore, crop], offset=offset, name="upscore")
  # upscore = mx.symbol.Crop(*[input, crop], offset=offset, name="upscore")
  softmax = mx.symbol.SoftmaxOutput(
      data=upscore, multi_output=True, use_ignore=True, ignore_label=255, name="softmax")
  return softmax


def get_fcn32s_symbol(numclass=21, workspace_default=1024):
  data = mx.symbol.Variable(name="data")
  pool3 = vgg16_pool3(data, workspace_default)
  pool4 = vgg16_pool4(pool3, workspace_default)
  score = vgg16_score(pool4, numclass, workspace_default)
  softmax = fcnxs_score(score, data, offset()[
                        "fcn32s_upscore"], (64, 64), (32, 32), numclass, workspace_default)
  return softmax


def get_fcn16s_symbol(numclass=21, workspace_default=1024):
  data = mx.symbol.Variable(name="data")
  pool3 = vgg16_pool3(data, workspace_default)
  pool4 = vgg16_pool4(pool3, workspace_default)
  score = vgg16_score(pool4, numclass, workspace_default)
  # score 2X
  score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2), num_filter=numclass,
                                   adj=(1, 1), workspace=workspace_default, name="score2")  # 2X
  score_pool4 = mx.symbol.Convolution(data=pool4, kernel=(1, 1), num_filter=numclass,
                                      workspace=workspace_default, name="score_pool4")
  score_pool4c = mx.symbol.Crop(
      *[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
  score_fused = score2 + score_pool4c
  softmax = fcnxs_score(score_fused, data, offset()[
                        "fcn16s_upscore"], (32, 32), (16, 16), numclass, workspace_default)
  return softmax


def get_fcn8s_symbol(numclass=21, workspace_default=1024):
  data = mx.symbol.Variable(name="data")
  pool3 = vgg16_pool3(data, workspace_default)
  pool4 = vgg16_pool4(pool3, workspace_default)
  score = vgg16_score(pool4, numclass, workspace_default)
  # score 2X
  score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2), num_filter=numclass,
                                   adj=(1, 1), workspace=workspace_default, name="score2")  # 2X
  score_pool4 = mx.symbol.Convolution(data=pool4, kernel=(1, 1), num_filter=numclass,
                                      workspace=workspace_default, name="score_pool4")
  score_pool4c = mx.symbol.Crop(
      *[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
  score_fused = score2 + score_pool4c
  # score 4X
  score4 = mx.symbol.Deconvolution(data=score_fused, kernel=(4, 4), stride=(2, 2), num_filter=numclass,
                                   adj=(1, 1), workspace=workspace_default, name="score4")  # 4X
  score_pool3 = mx.symbol.Convolution(data=pool3, kernel=(1, 1), num_filter=numclass,
                                      workspace=workspace_default, name="score_pool3")
  score_pool3c = mx.symbol.Crop(
      *[score_pool3, score4], offset=offset()["score_pool3c"], name="score_pool3c")
  score_final = score4 + score_pool3c
  softmax = fcnxs_score(score_final, data, offset()[
                        "fcn8s_upscore"], (16, 16), (8, 8), numclass, workspace_default)
  return softmax


# make a bilinear interpolation kernel, return a numpy.ndarray
def upsample_filt(size):
  factor = (size + 1) // 2
  if size % 2 == 1:
    center = factor - 1.0
  else:
    center = factor - 0.5
  og = np.ogrid[:size, :size]
  return (1 - abs(og[0] - center) / factor) * \
      (1 - abs(og[1] - center) / factor)


def init_from_vgg16(ctx, fcnxs_symbol, vgg16fc_args, vgg16fc_auxs):
  fcnxs_args = vgg16fc_args.copy()
  fcnxs_auxs = vgg16fc_auxs.copy()
  for k, v in fcnxs_args.items():
    if(v.context != ctx):
      fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
      v.copyto(fcnxs_args[k])
  for k, v in fcnxs_auxs.items():
    if(v.context != ctx):
      fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
      v.copyto(fcnxs_auxs[k])
  data_shape = (1, 3, 500, 500)
  arg_names = fcnxs_symbol.list_arguments()
  arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
  rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                      if x[0] in ['score_weight', 'score_bias', 'score_pool4_weight', 'score_pool4_bias',
                                  'score_pool3_weight', 'score_pool3_bias']])
  fcnxs_args.update(rest_params)
  deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
                        if x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight']])
  for k, v in deconv_params.items():
    filt = upsample_filt(v[3])
    initw = np.zeros(v)
    # becareful here is the slice assing
    initw[range(v[0]), range(v[1]), :, :] = filt
    fcnxs_args[k] = mx.nd.array(initw, ctx)
  return fcnxs_args, fcnxs_auxs


def init_from_fcnxs(ctx, fcnxs_symbol, fcnxs_args_from, fcnxs_auxs_from):
  """ use zero initialization for better convergence, because it tends to oputut 0,
  and the label 0 stands for background, which may occupy most size of one image.
  """
  fcnxs_args = fcnxs_args_from.copy()
  fcnxs_auxs = fcnxs_auxs_from.copy()
  for k, v in fcnxs_args.items():
    if(v.context != ctx):
      fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
      v.copyto(fcnxs_args[k])
  for k, v in fcnxs_auxs.items():
    if(v.context != ctx):
      fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
      v.copyto(fcnxs_auxs[k])
  data_shape = (1, 3, 500, 500)
  arg_names = fcnxs_symbol.list_arguments()
  arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
  rest_params = {}
  deconv_params = {}
  # this is fcn8s init from fcn16s
  if 'score_pool3_weight' in arg_names:
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                        if x[0] in ['score_pool3_bias', 'score_pool3_weight']])
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0]
                          in ["bigscore_weight", 'score4_weight']])
  # this is fcn16s init from fcn32s
  elif 'score_pool4_weight' in arg_names:
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                        if x[0] in ['score_pool4_weight', 'score_pool4_bias']])
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0]
                          in ["bigscore_weight", 'score2_weight']])
  # this is fcn32s init
  else:
    logging.error(
        "you are init the fcn32s model, so you should use init_from_vgg16()")
    sys.exit()
  fcnxs_args.update(rest_params)
  for k, v in deconv_params.items():
    filt = upsample_filt(v[3])
    initw = np.zeros(v)
    # becareful here is the slice assing
    initw[range(v[0]), range(v[1]), :, :] = filt
    fcnxs_args[k] = mx.nd.array(initw, ctx)
  return fcnxs_args, fcnxs_auxs


def main():
  fcnxs = get_fcn32s_symbol(numclass=21, workspace_default=1536)
  fcnxs_model_prefix = "model_pascal/FCN32s_VGG16"
  if args.model == "fcn16s":
    fcnxs = get_fcn16s_symbol(numclass=21, workspace_default=1536)
    fcnxs_model_prefix = "model_pascal/FCN16s_VGG16"
  elif args.model == "fcn8s":
    fcnxs = get_fcn8s_symbol(numclass=21, workspace_default=1536)
    fcnxs_model_prefix = "model_pascal/FCN8s_VGG16"
  arg_names = fcnxs.list_arguments()
  _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(args.prefix, args.epoch)
  if not args.retrain:
    if args.init_type == "vgg16":
      fcnxs_args, fcnxs_auxs = init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    elif args.init_type == "fcnxs":
      fcnxs_args, fcnxs_auxs = init_from_fcnxs(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
  train_dataiter = FileIter(
      root_dir="./data/simple/",
      meta="./data/meta/part_train.json",
      # cut_off_size         = 400,
      # rgb_mean             = (123.68, 116.779, 103.939),
      mu=29.5565373846,
      sigma=63.7321981505
  )
  val_dataiter = FileIter(
      root_dir="./data/simple/",
      meta="./data/meta/part_val.json",
      # rgb_mean             = (123.68, 116.779, 103.939),
      mu=29.5565373846,
      sigma=63.7321981505
  )
  model = Solver(
      ctx=ctx,
      symbol=fcnxs,
      begin_epoch=0,
      num_epoch=50,
      arg_params=fcnxs_args,
      aux_params=fcnxs_auxs,
      learning_rate=1e-10,
      momentum=0.99,
      wd=0.0005)
  model.fit(
      train_data=train_dataiter,
      eval_data=val_dataiter,
      batch_end_callback=mx.callback.Speedometer(1, 10),
      epoch_end_callback=mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description='Convert vgg16 model to vgg16fc model.')
  parser.add_argument('--model', default='fcn8s',
                      help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
  parser.add_argument('--prefix', default='VGG_FC_ILSVRC_16_layers',
                      help='The prefix(include path) of vgg16 model with mxnet format.')
  parser.add_argument('--epoch', type=int, default=74,
                      help='The epoch number of vgg16 model.')
  parser.add_argument('--init-type', default="vgg16",
                      help='the init type of fcn-xs model, e.g. vgg16, fcnxs')
  parser.add_argument('--retrain', action='store_true', default=False,
                      help='true means continue training.')
  args = parser.parse_args()
  logging.info(args)
  main()
