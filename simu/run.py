# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-08-27 10:27:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-28 00:47:29
import numpy as np
import ctypes
import argparse
import logging
import subprocess
import os

from gen import gen_exp, get_input
from robust_space_learning import robust_space_alignment

def init():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dat", help = "", type = str)
  parser.add_argument("--job", help = "", type = str)
  parser.add_argument("--threads", help = "", type = int)
  args = parser.parse_args()

  head = '%(asctime)s %(message)s'
  logging.basicConfig(level=logging.DEBUG, format = head)
  logging.getLogger().addHandler(logging.FileHandler("simu.log"))

  mkl_rt = ctypes.CDLL('libmkl_rt.so')
  mkl_rt.MKL_Set_Num_Threads(args.threads)
  coreset = mkl_rt.MKL_Get_Max_Threads()
  logging.info('use maximum %d threads for computing.' % coreset)
  logging.info('data path %s...' % args.dat)

  return args

def main():
  args = init()
  if args.job == 'gen':
    gen_exp()
  elif args.job == 'recg':
    output = subprocess.check_output('')
  elif args.job == 'splearn':
    Is = get_input(args.dat)
    T, A, E, Ts, D, converge, t, in_iter, out_iter = robust_space_alignment(Is)

if __name__ == '__main__':
  main()
