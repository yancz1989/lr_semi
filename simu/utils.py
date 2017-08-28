# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-08-27 10:33:57
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-08-27 11:07:41

import os
import os.path
import logging

def mkdir(dir):
  if not os.path.exists(dir):
    os.mkdirs(dir)

