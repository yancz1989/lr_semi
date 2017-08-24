# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2017-05-08 11:24:49
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-05-15 10:12:20

import logging
import argparse
import os
import os.path
import json
import collections
import shutil

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)