#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2015-11-09 19:07:28
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-30 00:22:22

import numpy as np
import scipy as sp
from numpy import linalg as LA
import scipy.sparse.linalg as sLA
from PIL import Image
import time

def meta(I, Isize):
  Iy, Ix = np.gradient(I)
  U = [j + 1 for i in range(Isize[0]) for j in range(Isize[1])]
  V = [i + 1 for i in range(Isize[0]) for j in range(Isize[1])]
  return (I.flatten(), Ix, Iy, np.array(U), np.array(V))

def jacobian(I0, Ix0, Iy0, U, V, T, Isize):
  Iwarp = np.array(Image.fromarray(I0).transform(
    (Isize[1], Isize[0]), Image.PERSPECTIVE, T.ravel()[:8], Image.BICUBIC))
  Ix = np.array(Image.fromarray(Ix0).transform(
    (Isize[1], Isize[0]), Image.PERSPECTIVE, T.ravel()[:8], Image.BICUBIC)).flatten()
  Iy = np.array(Image.fromarray(Iy0).transform(
    (Isize[1], Isize[0]), Image.PERSPECTIVE, T.ravel()[:8], Image.BICUBIC)).flatten()

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
  return J, I

def thresholding(x, thr):
  return np.maximum(x - thr, 0)

def robust_space_alignment(Is, maxi_out = 100, maxi_in = 1000):
  n = Is.shape[0]
  h = Is.shape[1]
  w = Is.shape[2]
  D, Ad, Ed, Ixs, Iys = [np.zeros((w * h, n)),
      np.zeros((w * h, n)), np.zeros((w * h, n)),
      np.zeros((n, h, w)), np.zeros((n, h, w))]
  Us, Vs = np.zeros((n, (h * w))), np.zeros((n, (h * w)))
  Qs, Rs = np.zeros((n, (w * h), 8)), np.zeros((n, 8, 8))
  Ts, dXs = np.zeros((n, 3, 3)), np.zeros((n, 8))
  Js = np.zeros((n, (w * h), 8))

  for i, I in enumerate(Is):
    D[:, i], Ixs[i], Iys[i], Us[i], Vs[i] = meta(
            I.astype(np.float64), (h, w))
    Ts[i] = np.eye(3)

  out_iter = 0
  in_iter = 0

  rho = 1.25
  lmbd = 1.0 / 100
  cur = 0
  pre = 1.0e20
  converage = 0

  tol = 1e-7

  # outer loop
  st = time.clock()
  while out_iter < maxi_out:
    out_iter += 1
    for i in range(n):
      Js[i], D[:, i] = jacobian(Is[i], Ixs[i], Iys[i], Us[i], Vs[i], Ts[i], (h, w))
      Qs[i], Rs[i] = np.linalg.qr(Js[i])
    lmbd = 1. / np.sqrt(w * h)

    # get D = A + E and delta_T
    iit = 1
    Y = D
    norm2 = np.linalg.norm(Y, ord = 2)
    norm_inf = np.linalg.norm(Y.flatten(), np.inf) / lmbd
    dn = max(norm2, norm_inf)
    Y = Y / dn

    normf = np.linalg.norm(Y, ord = 'fro')
    Ad, Ed, dt, dtm, mu, rho = (np.zeros(D.shape), np.zeros(D.shape), 
        np.zeros((n, 8)), np.zeros(D.shape), 1.25 / normf, 1.25)

    while iit < maxi_in:
      iit += 1

      tmp = D + dtm - Ed + (1. / mu) * Y
      # U, S, V = sLA.svds(tmp, k = 10)
      U, S, V = np.linalg.svd(tmp, full_matrices = False)
      Shat = thresholding(S, 1. / mu)
      Ad = U.dot(np.diag(Shat).dot(V))

      tmp = D + dtm - Ad + 1. / mu * Y
      Ed = np.sign(tmp) * np.maximum(np.abs(tmp) - lmbd / mu, 0.)

      tmp = D - Ed - Ad + 1. / mu * Y
      for i in range(n):
        dt[i, :] = -Qs[i].T.dot(tmp[:, i]).T
        dtm[:, i] = Qs[i].dot(dt[i, :])

      Z = D + dtm - Ad - Ed
      Y = Y + mu * Z
      mu = mu * rho

      nuc = np.sum(Shat)
      l1 =  np.sum(np.abs(Ed))
      obj = nuc + l1
      print('nuc %f l1 %f' % (nuc, l1))
      if np.linalg.norm(Z, 'fro') / normf < tol:
        print('stop after inner iterations %d, at obj value %f.' % (
          iit, obj))
      elif iit % 5 == 0:
        print('inner iteration %d, objv %f' % (iit, obj))

    in_iter += iit
    for i in range(n):
      Ts[i].ravel()[:8] = Ts[i].ravel()[:8] + np.linalg.inv(Rs[i]).dot(dXs[i])
    print('Out iteration, obj value %f.' % obj)
    if pre - obj < 1e-5:
      print('previous obj %f, diff %f, stopped.' % (pre, obj))
      break
    pre = obj

  return (T, Ad, Ed, Ts, D, converage,
              time.clock() - st, in_iter, out_iter)

