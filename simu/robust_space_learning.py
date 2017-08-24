#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2015-11-09 19:07:28
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-22 08:37:02

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-02-26 09:12:16
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-03-14 21:47:44

import numpy as np
import scipy as sp
import cv2
from numpy import linalg as LA
from util import *
from scipy import io as sio
from PIL import Image
import time

# def meta(I0, coords, Isize, As):
#     Iy0, Ix0 = np.gradient(I0)
#     idx = []
#     mpX = [0]
#     mpE = [0]
#     U = []
#     V = []

#     for (A, coor) in zip(As, coords):
#         idx += [i * Isize[1] + j for i in range(coor[2], coor[3]) for j in range(coor[0], coor[1])]
#         mpX.append(A.shape[1])
#         mpE.append(len(idx))
#         U += [j + 1 for i in range(coor[2], coor[3]) for j in range(coor[0], coor[1])]
#         V += [i + 1 for i in range(coor[2], coor[3]) for j in range(coor[0], coor[1])]
#     for i in range(len(mpX) - 1):
#         mpX[i + 1] = mpX[i] + mpX[i + 1]
#     return (Ix0, Iy0, np.array(idx), np.array(U), np.array(V), mpX, mpE)

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
  Ts, dXs = np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
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
    while iit < maxi_in:
      iit += 1


    in_iter += iit
    for i in range(n):
      Ts[i] = Ts[i] + 



  while out_iter < maxi_out:
    out_iter += 1
    Ix, Iy, I, J, Inorm = jacobian(I0, Ix0, Iy0, idx, U, V, T, Isize, mpE)
    # print J
    Q, R = LA.qr(J)
    # inner loop
    mu = 1.25 / LA.norm(I)
    dual_norm = max(LA.norm(I), LA.norm(I, np.inf) / lmbd)

    Y = I / dual_norm
    iiter = 0
    e = np.zeros(len(idx))
    Ax = np.hstack(np.array([A[:, 0] for A in As]))
    while iiter < maxi_in:
      iiter += 1
      Jdltau = -Q.T.dot(I - e - Ax + 1 / mu * Y)
      Jm = Q.dot(Jdltau)
      tmp = I + Jm - Ax + 1 / mu * Y
      e = np.sign(tmp) * (np.abs(tmp) > (lmbd / mu)) * (np.abs(tmp) - lmbd / mu)
      tmp = I + Jm - e + 1 / mu * Y
      for (i, IA, A) in zip(range(len(As)), IAs, As):
        iX = range(mpX[i], mpX[i + 1])
        iE = range(mpE[i], mpE[i + 1])
        x[iX] = IA.dot(tmp[iE])
        Ax[iE] = A.dot(x[iX])
      Z = I + Jm - Ax - e
      Y = Y + mu * Z
      mu = mu * rho
      if LA.norm(Z) / LA.norm(I) < 1e-5:
        break
        
    in_iter += iiter
    delta_tau = LA.pinv(R).dot(Jdltau)
    tau = tau + delta_tau
    T.ravel()[0 : 8] = tau
    # print T
    if abs(LA.norm(e, 1) - pre) < 1e-3:
      converage = 1
      break;
    else:
      pre = LA.norm(e, 1)
  return (T, converage, time.clock() - st, in_iter, out_iter)

