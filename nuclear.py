# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-05-08 11:23:48
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-17 01:04:30
import os
import mxnet as mx
import numpy as np
import scipy as sp
from scipy.sparse.linalg import svds
import logging

class Nuclear(mx.operator.CustomOp):
  def __init__(self):
    self.cnt = -1
    self.A = np.zeros((16, 224 * 396))
    self.E = np.zeros((16, 224 * 396))
    self.Y = np.zeros((16, 224 * 396))
    self.alpha = 1
    self.mu = 1e-2
    self.lmbd = self.mu / 4
    self.gamma = 1. / 50000.

  def forward(self, is_train, req, in_data, out_data, aux):
    x = in_data[0].asnumpy()
    sfmx = np.zeros(x.shape)
    for k in range(x.shape[0]):
      sfmx[k, :, :, :] = np.exp(x[k, :, :, :] - x[k, :, :, :].max(axis=0))
      sfmx[k, :, :, :] /= sfmx[k, :, :, :].sum(axis=0)
    self.D = np.reshape(sfmx.argmax(axis = 1), (x.shape[0], x.shape[2] * x.shape[3]))

    self.assign(out_data[0], req[0], mx.nd.array(sfmx))


  def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    self.cnt = self.cnt + 1
    l = in_data[1].asnumpy()
    y = out_data[0].asnumpy()

    l = l.reshape(l.shape[0], l.shape[2], l.shape[3]).astype(np.int)
    y[np.arange(1)[:, None, None], l, np.arange(l.shape[1])[None, :, None], np.arange(l.shape[2])[None, None, :]] -= l

    if self.cnt % 5 == 0 or self.cnt < 10:
      self.assign(in_grad[0], req[0], mx.nd.array(y))
      print('iter %d, conv gradient: %f.' % (self.cnt, np.linalg.norm(y.flatten())))
    elif self.cnt % 5 == 1:
      # update conv with regularize
      tmp = (self.gamma * (self.D - self.A - self.E) + self.Y).reshape(16, 224, 396)
      shp = y.shape
      P = np.zeros(shp)
      idx = (tmp > 0).astype(int)
      P[np.arange(shp[0])[:, None, None], idx, np.arange(shp[2])[None, :, None], np.arange(shp[3])[None, None, :]] = tmp
      Q = self.alpha * y + P
      self.assign(in_grad[0], req[0], mx.nd.array(Q))
      print('iter %d, conv with regulariztion gradient: %f.' % (self.cnt, np.linalg.norm(Q.flatten())))
    elif self.cnt % 5 == 2:
      # update A
      P = self.Y + self.gamma * self.D - self.gamma * self.E
      U, S, V = svds(P, k = 4)
      self.A = U.dot(np.diag(S - self.mu / self.gamma)).dot(V)
      print('iter %d, nuclear component: %f.' % (self.cnt, np.sum(S)))
    elif self.cnt % 5 == 3:
      # update E
      tmp = self.D - self.A + self.Y
      self.E = np.maximum(tmp - self.lmbd, 0) + np.maximum(tmp + self.lmbd, 0)
      print('iter %d, sparse component: %f.' % (self.cnt, np.sum(self.E)))
    else:
      # update Y
      self.Y = self.Y + self.gamma * (self.D - self.E - self.A)
      self.gamma = self.gamma * 2


@mx.operator.register("Nuclear")
class NuclearProp(mx.operator.CustomOpProp):
  def __init__(self):
    super(NuclearProp, self).__init__(need_top_grad=False)

  def list_arguments(self):
    return ['data', 'label']

  def list_outputs(self):
    return ['output']

  def infer_shape(self, in_shape):
    data_shape = in_shape[0]
    label_shape = (in_shape[0][0], 1, data_shape[2], data_shape[3])
    output_shape = (in_shape[0][0], 16, data_shape[2], data_shape[3])
    return [data_shape, label_shape], [output_shape], []

  def infer_type(self, in_type):
    return [in_type[0], in_type[0]], [in_type[0]], []

  def create_operator(self, ctx, shapes, dtypes):
    return Nuclear()
