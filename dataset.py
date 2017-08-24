# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-05-08 15:17:58
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-16 20:37:03

from utils import mkdir
import json
import numpy as np
import cv2
import os
from mxnet.io import DataIter

META_ROOT = 'data/meta/'
NIST_ROOT = 'data/nist/'
SIMPLE_ROOT = 'data/simple/'
ROOT_ENTRIES = 'data/nist/sd02/tables/'

def make_meta_nist(root = NIST_ROOT, root_entries = ROOT_ENTRIES):
  datlst = {}
  entries = os.listdir(root_entries)
  for e in entries:
    datlst[e[:-4]] = []
  for path, _, fnames in os.walk(root):
    for f in fnames:
      if f[0] == '.':
        continue
      fname = path + '/' + f
      if f[-3:] == 'png' and f[0] != '.':
        with open(fname[:-3] + 'fmt') as fmt:
          title = fmt.readline()[:-1]
          datlst[title].append(fname)
  with open(META_ROOT + 'nist.json', 'w') as jf:
    json.dump(datlst, jf)

def make_noise(root, mu, sigma):
  def resize_img(img):
    scalar = 0.15
    shp = img.shape
    aff = (np.eye(3) * scalar)[:2, :]
    return cv2.warpAffine(img, aff, dsize = (int(shp[1] * scalar), int(shp[0] * scalar)))

  fcnt = 0
  for path, _, fnames in os.walk(root):
    for f in fnames:
      if f[-3:] == 'png' and f[0] != '.':
        img = cv2.imread(path + '/' + f, 0)
        seg = 255 - img[100:550, :]
        input = (np.random.randn(*seg.shape) * sigma + mu) + seg

        seg[np.where(seg < 1.0)] = 0
        seg[np.where(seg > 1.0)] = 1

        cv2.imwrite(SIMPLE_ROOT + 'seg/' + f[:-3] + 'bmp', seg)
        cv2.imwrite(SIMPLE_ROOT + 'input/' + f[:-3] + 'bmp', input)
        fcnt += 1
        if fcnt % 20 == 0:
          print('finished %d images...' % fcnt)

class FileIter(DataIter):
  """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
  in this example, we use the whole image training for fcn-xs, that is to say
  we do not need resize/crop the image to the same size, so the batch_size is
  set to 1 here
  Parameters
  ----------
  root_dir : string
      the root dir of image/label lie in
  flist_name : string
      the list file of iamge and label, every line owns the form:
      index \t image_data_path \t image_label_path
  cut_off_size : int
      if the maximal size of one image is larger than cut_off_size, then it will
      crop the image with the minimal size of that image
  data_name : string
      the data name used in symbol data(default data name)
  label_name : string
      the label name used in symbol softmax_label(default label name)
  """
  def __init__(self, root_dir, meta,
               # rgb_mean = (117, 117, 117),
               mu = 127.0,
               sigma = 30.0,
               cut_off_size = None,
               data_name = "data",
               label_name = "Nuclear_label"):
    super(FileIter, self).__init__()
    self.root_dir = root_dir
    self.flist_name = meta
    # self.mean = np.array(rgb_mean)  # (R, G, B)
    self.mu = mu
    self.sigma = sigma
    self.cut_off_size = cut_off_size
    self.data_name = data_name
    self.label_name = label_name

    self.f = json.load(open(self.flist_name, 'r'))
    self.num_data = len(self.f)
    self.cursor = 0
    self.data, self.label = self._read()

  def _read(self):
    """get two list, each list contains two elements: name and nd.array value"""
    # _, data_img_name, label_img_name = self.f.readline().strip('\n').split("\t")
    fname = self.f[self.cursor]
    data = {}
    label = {}
    print(self.label_name)
    data[self.data_name], label[self.label_name] = self._read_img(fname, fname)
    return list(data.items()), list(label.items())

  def _read_img(self, img_name, label_name):
    img = cv2.imread(os.path.join(self.root_dir, 'input', img_name),
                      cv2.IMREAD_COLOR).astype(np.float32)
    label = cv2.imread(os.path.join(self.root_dir, 'seg', img_name), 0).astype(np.int32)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    # print(img.shape)

    # assert img.size == label.size
    if self.cut_off_size is not None:
      max_hw = max(img.shape[0], img.shape[1])
      min_hw = min(img.shape[0], img.shape[1])
      if min_hw > self.cut_off_size:
        rand_start_max = round(np.random.uniform(0, max_hw - self.cut_off_size - 1))
        rand_start_min = round(np.random.uniform(0, min_hw - self.cut_off_size - 1))
        if img.shape[0] == max_hw :
          img = img[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
          label = label[rand_start_max : rand_start_max + self.cut_off_size, rand_start_min : rand_start_min + self.cut_off_size]
        else :
          img = img[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
          label = label[rand_start_min : rand_start_min + self.cut_off_size, rand_start_max : rand_start_max + self.cut_off_size]
      elif max_hw > self.cut_off_size:
        rand_start = round(np.random.uniform(0, max_hw - min_hw - 1))
        if img.shape[0] == max_hw :
          img = img[rand_start : rand_start + min_hw, :]
          label = label[rand_start : rand_start + min_hw, :]
        else :
          img = img[:, rand_start : rand_start + min_hw]
          label = label[:, rand_start : rand_start + min_hw]
    # reshaped_mean = self.mean.reshape(1, 1, 3)
    img = (img - self.mu) / self.sigma
    # print(img.shape)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # (c, h, w)
    img = np.expand_dims(img, axis=0)  # (1, c, h, w)
    # label =  # (h, w)
    label = np.expand_dims(label, axis=0)  # (1, h, w)
    return (img, label)

  @property
  def provide_data(self):
    """The name and shape of data provided by this iterator"""
    return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]

  @property
  def provide_label(self):
    """The name and shape of label provided by this iterator"""
    return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]

  def get_batch_size(self):
    return 1

  def reset(self):
    self.cursor = -1

  def iter_next(self):
    self.cursor += 1
    if(self.cursor < self.num_data - 1):
      return True
    else:
      return False

  def next(self):
    """return one dict which contains "data" and "label" """
    if self.iter_next():
      self.data, self.label = self._read()
      return {self.data_name  :  self.data[0][1],
              self.label_name :  self.label[0][1]}
    else:
      raise StopIteration


def main():
  make_noise(SIMPLE_ROOT + 'resized/', .0, 30.0)
  # make_meta_nist()

if __name__ == '__main__':
  main()
