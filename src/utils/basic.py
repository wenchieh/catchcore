#!/usr/bin/python
# -*- coding=utf-8 -*-

#  Project: catchcore
#    basic.py
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <10/23/2018>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <10/23/2018>
#

__author__ = 'wenchieh'

# sys
import math
import collections as clct

# third-part lib
import numpy as np

# some constant variable settings
MAXVAL = 1.0
cF = 8  # 32 # bit
ZERO = 1.e-10
INF = 1.e+10


def log_2(x):
    if x == 0:
        return 0
    return np.log2(x)


def log_s(x):
    if x == 0:
        return 0
    return 2 * log_2(x) + 1.0


def vector_huffman(vector):
    code = 0.0
    ns = len(vector) * 1.0
    k2v = clct.Counter(vector)
    for k in k2v.values():
        code += (k / ns) * log_2(k / ns)
    return -ns * code


def normpdf(x, mean, std):
    var = float(std) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(- (float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom
