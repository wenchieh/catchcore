#!/usr/bin/python
# -*- coding=utf-8 -*-

#  Project: catchcore
#    toolz.py
#          Some basic functions as tool.
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <10/30/2018>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <10/30/2018>
#

__author__ = 'wenchieh'


# third party lib
import numpy as np

# project
from .utils.ioutils import load_tensor
from .tailorten import TailorTen


def gen_evendense_blocks(p, size, base=None):
    blocks = np.random.binomial(1, p, size)
    subs = np.nonzero(blocks)
    vals = blocks[subs]
    if base is not None:
        subs = list(subs)
        for k in range(len(size)):
            subs[k] += base[k]
    return tuple(subs), vals


def gen_hierdense_blocks(ps, sizes, scales=None, base=None):
    hs = len(ps)
    res = dict()
    if scales is None:
        scales = np.ones((hs, ), int)
    for h in range(hs):
        hsub, hval = gen_evendense_blocks(ps[h], sizes[h], base)
        if len(res) == 0:
            res = dict(zip(map(tuple, np.asarray(hsub).T), scales[h] * hval))
        else:
            res = dict(dict(zip(map(tuple, np.asarray(hsub).T), scales[h] * hval)), **res)
    return np.array(res.keys(), int), np.array(res.values(), int)


def initialize_tailortens(infn, valcol, labelcol, valtype=int, labeltype=int,
                          usecols=None, sep=',', comments='%'):
    subs, vals, labels = load_tensor(infn, valcol, valtype, labelcol, labeltype, sep, comments)
    if usecols is not None:
        subs = subs[:, np.asarray(usecols)]
    tten = TailorTen(subs, vals, accum_fun=sum.__call__)

    return tten, labels
