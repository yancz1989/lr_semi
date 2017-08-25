#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2015-11-09 19:07:28
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-25 17:19:40

import numpy as np
import scipy as sp
import cv2
from numpy import linalg as LA
from util import *
from scipy import io as sio
from PIL import Image
import time

def meta(I, Isize):
  Iy, Ix = np.gradient(I)
  U = [j + 1 for i in range(Isize[0]) for j in range(Isize[1])]
  V = [i + 1 for i in range(Isize[0]) for j in range(Isize[1])]
  return (I.flatten(), Ix, Iy, np.array(U), np.array(V))

def jacobian(I0, Ix0, Iy0, U, V, T, Isize):
  T_ = LA.inv(T)
  Iwarp = cv2.warpPerspective(I0, T_, (Isize[1], Isize[0]), borderValue = 0, flags = cv2.INTER_CUBIC)
  Ix = cv2.warpPerspective(Ix0, T_, (Isize[1], Isize[0]), borderValue = 255, flags = cv2.INTER_CUBIC).flatten()
  Iy = cv2.warpPerspective(Iy0, T_, (Isize[1], Isize[0]), borderValue = 255, flags = cv2.INTER_CUBIC).flatten()

  I = Iwarp.flatten()
  Inorm = LA.norm(I)
  Ix = Ix / Inorm - sum(Ix * I) / (Inorm ** 3) * I
  Iy = Iy / Inorm - sum(Iy * I) / (Inorm ** 3) * I
  I = I / Inorm

  X = U * T[0, 0] + V * T[0, 1] + T[0, 2]
  Y = U * T[1, 0] + V * T[1, 1] + T[1, 2]
  P = U * T[2, 0] + V * T[2, 1] + 1
  J = np.vstack(( Ix * U / P, Ix * V / P, Ix / P,
          Iy * U / P, Iy * V / P, Iy / P,
          -Ix * X * U / (P ** 2) - Iy * Y * U / (P ** 2),
          -Ix * X * V / (P ** 2) - Iy * Y * V / (P ** 2) )).T
  return J

def robust_space_alignment(Is, maxi_out = 100, maxi_in = 1000):
  n = Is.shape[0]
  h = Is.shape[1]
  w = Is.shape[2]
  D, A, E, Ad, Ed, Ixs, Iys = [np.zeros((w * h, n)),
      np.zeros((w * h, n)), np.zeros((w * h, n)),
      np.zeros((w * h, n)), np.zeros((w * h, n)),
      np.zeros((n, w, h)), np.zeros((n, w, h))]
  Us, Vs = np.zeros((n, w, h)), np.zeros((n, w, h))
  Qs, Rs = np.zeros((n, (w * h), 8)), np.zeros((n, 8, 8))
  Ts, dXs = np.zeros((n, 3, 3)), np.zeros((n, 8))
  Js = np.zeros(n, (w * h), 8)

  for i, I in enumerate(Is):
    D[:, i], Ixs[i], Iys[i], Us[i], V[i] = meta(
            I.astype(np.float64), Isize)
    T[i] = np.eye(3)

  out_iter = 0
  in_iter = 0

  rho = 1.25
  lmbd = 1.0 / 100
  cur = 0
  IAs = np.array([LA.pinv(A) for A in As])
  x = np.zeros(np.sum([A.shape[1] for A in As]))
  pre = 1.0e20
  converage = 0

  tol = 1e-7

  # outer loop
  st = time.clock()
  while out_iter < maxi_out:
    out_iter += 1
    for i in range(n):
      Js[i] = jacobian(Is[i], Ixs[i], Iys[i], Us[i], Vs[i], Ts[i], (h, w))
      Qs[i], Rs[i] = np.linalg.qr(Js[i])
    lmbd = 1 / np.sqrt(w * h)

    # get D = A + E and delta_T
    iit = 1
    Y = D
    norm2 = 
    while iit < maxi_in:
      iit += 1


    in_iter += iit
    for i in range(n):
      Ts[i].ravel()[:8] = Ts[i].ravel()[:8] + np.linalg.inv(Rs[i]).dot(dXs[i])


  return (T, converage, time.clock() - st, in_iter, out_iter)

