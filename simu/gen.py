# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-20 22:36:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-28 00:47:13

import os
import os.path
import shutil
import json
from json import encoder
import itertools
import logging

import numpy as np
import scipy as sp

import PIL
from PIL import ImageDraw, Image, ImageFont

def get_input(root):
  Is = np.array([np.array(Image.open(root + '/' + f)) for f in os.listdir(root) if (not f[0] == '.' and f[-3:] == 'bmp')])
  print(Is.shape)
  logging.info('date size (%d, %d, %d).' % (Is.shape[0], Is.shape[1], Is.shape[2]))
  return Is

def rtransform(pos, rangeA, rangeP):
  def affM(idx, val):
    M = np.eye(3)
    for i, j in enumerate(idx):
      M[j // 3, j % 3] = val[i]
    return M
  
  def rotate(M, theta):
    return affM([0, 1, 3, 4], [np.cos(theta),
      np.sin(theta), -np.sin(theta), np.cos(theta)]).dot(M)
  
  def scale(M, alpha):
    return affM([0, 4], [alpha for i in range(2)]).dot(M)
  
  def translate(M, T):
    return affM([2, 5], [T[0], T[1]]).dot(M)
  
  def rotateC(M, theta, shp):
    return translate(rotate(translate(M,
        [-shp[0] / 2, -shp[1] / 2]), theta), [shp[0] / 2, shp[1] / 2])
  
  def scaleC(M, alpha, shp):
    return translate(scale(translate(M, [-shp[0] / 2, -shp[1] / 2]), alpha),
          [shp[0] / 2, shp[1] / 2])
  
  def perspective(M, P):
    return affM([6, 7], [P[0], P[1]]).dot(M)

  P = (np.random.rand(2) - .5) * 2 * rangeP
  theta = (-1. if np.random.randint(2) == 0 else 1.) * np.random.rand() * rangeA * np.pi / 180.
  return translate(perspective(rotateC(affM([], []), theta, (600, 400)), P), np.random.randint(47, size = (2,)) - 23)

def get_sample(format):
  acc = format['acc']
  def rand_recg(acc, k):
    return np.random.randint(255) if np.random.rand() > acc else k
  chars = []
  for k, dt in enumerate(format['bks']):
    fs = dt[0]
    fix = dt[1]
    flex = np.random.randint(dt[2] + 1)
    x = dt[3]
    y = dt[4]
    sep = dt[5]
    for i in range(fix):
      chars.append(
        [rand_recg(acc, k), x + fs // 2 + fs * i, y + fs // 2])
    for i in range(flex):
      chars.append([np.random.randint(255),
        x + fs * (fix + i) + fs // 2, y + fs // 2])
  return chars

def gen_format(style, rA, rP, acc, idx):
  format = {}
  format['P'] = rP
  format['A'] = rA
  format['acc'] = acc
  format['name'] = '%d_%d' % (style, idx)
  if style == 0:
    format['bks'] = [
                      [30, 10, 2, 100, 50, 40],
                      [25, 5, 8, 90, 130, 50],
                      [40, 5, 3, 100, 220, 60],
                      [35, 6, 2, 100, 280, 0]
                    ]
  elif style == 1:
    format['bks'] = [
                      [30, 4, 6, 100, 50, 45],
                      [25, 8, 8, 150, 170, 45],
                      [40, 6, 4, 100, 220, 60],
                      [30, 7, 8, 100, 280, 0]
                    ]
  else:
    format['lu'] = [0, 120]
    format['bks'] = [
                      [40, 8, 0, 200, 80, 30],
                      [30, 2, 4, 50, 150, 30],
                      [30, 2, 2, 280, 150, 30],
                      [35, 4, 3, 200, 210, 40],
                      [25, 3, 7, 70, 285, 0]
                    ]
  return format

def do_draw(chars, para, r):
  img = Image.new('L', (600, 400), color = 'white')
  draw = ImageDraw.Draw(img)
  tchars = []
  for k in chars:
    p = np.zeros((3, 1))
    p[:2, 0] = k[1:]
    p[2] = 1.
    p = para.dot(p)
    p[:2] = p[:2] / p[2]
    draw.ellipse(
      (p[0] - r, p[1] - r, p[0] + r, p[1] + r),
      fill = 'black', outline = 'black')
    tchars.append([k[0], p[0, 0], p[1, 0]])
  return img, tchars

def gen_exp():
  style = [0, 1, 2]
  Arange = [0., 3., 5., 10.]
  Prange = [0., 1e-4]
  acc = [1., .9, .8, .7]
  for k, K in enumerate(itertools.product(style, Arange, Prange, acc)):
    format = gen_format(K[0], K[1], K[2], K[3], k)
    if not os.path.exists('dat/imgs/' + format['name']):
      os.makedirs('dat/imgs/' + format['name'])
    with open('dat/config/' + format['name'] + '.json', 'w') as f:
      json.dump(format, f)
    for i in range(1000):
      chars = get_sample(format)
      Ppara = rtransform([0, 0], K[1], K[2])
      img, tchars = do_draw(chars, Ppara, 10)
      jsobj = {}
      jsobj['org'] = chars
      jsobj['trans'] = tchars
      with open('dat/imgs/' + format['name'] + '/%03d.json' % i, 'w') as f:
        json.dump(jsobj, f)
      img.resize((60, 40), PIL.Image.BILINEAR).save(
          'dat/imgs/' + format['name'] + '/%03d.bmp' % i, 'BMP')

def main():
  np.random.seed(2012310818)
  gen_exp()

if __name__ == '__main__':
  main()

