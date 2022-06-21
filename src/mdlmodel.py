#!/usr/bin/python
# -*- coding=utf-8 -*-

# #  MDLModel
#  Author: wenchieh
#
#  Project: catchcore
#      mdlmodel.py:
#               The minimum description length (MDL) metric for the
#               resultant hierarchical dense subtensor
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <11/18/2018>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <11/18/2018>
#


__author__ = 'wenchieh'

# sys
from enum import Enum
from collections import Counter

# third-part libs
from scipy.stats import poisson

# project
from .utils.basic import *


class ProbModel(Enum):
    BINOMIALS = 1
    GAUSSIAN = 2
    POISSION = 3


def binomials_encode(vector, vols):
    ent2val = Counter(vector)
    if 0 not in ent2val:
        ent2val[0] = vols - len(vector)

    codes = log_s(ent2val[1])
    ns = vols * 1.0
    codes += ent2val[0] * log_2(ns / ent2val[0]) + ent2val[1] * log_2(ns / ent2val[1])

    return codes


def gaussian_encode(vector, vols):
    vector = np.asarray(vector)
    mean = np.sum(vector) / vols
    std = np.std(list(vector) + [0] * (vols - len(vector)))

    codes = log_s(vols)
    codemean, codestd = cF, cF
    if mean > 1:
        codemean += log_2(int(mean))
    if std > 1:
        codestd += log_2(int(std))
    codes += codemean + codestd

    neglogll = 0.5 * vols * (log_2(2 * np.pi * std ** 2) / 2 + 1.0 / np.log(2))
    codes += neglogll
    if neglogll >= 0:
        print("Warning: the code length for log likelihood is non-positive: {}".format(neglogll))
    return codes


def poisson_encode(vector, vols):
    vector = np.asarray(vector)
    lamb = np.sum(vector) / vols
    ent2cnt = Counter(vector)
    if 0 not in ent2cnt:
        ent2cnt[0] = vols - len(vector)

    codes = log_s(vols)
    codes += log_2(lamb) + cF
    rv = poisson(lamb)
    for e, cnt in ent2cnt.items():
        p_e = rv.pmf(e)
        codes += - cnt * log_2(p_e)
    return codes


class MDLModel(object):
    ten_ndim = 0
    ten_sub2val = None
    nhs = 0
    hr_idvs_col = None
    ten_shape = None
    hr_shape = None
    block_density_prob = None

    def __init__(self, ten_sub2val, hrivsc=None):
        self.ten_sub2val = ten_sub2val
        ten_ndim = len(list(self.ten_sub2val.keys())[0])
        pos_arr = np.asarray(list(ten_sub2val.keys()))
        ten_shape = [len(np.unique(pos_arr[:, d])) for d in range(ten_ndim)]

        if hrivsc is None or len(hrivsc) <= 0:
            self.ten_ndim = ten_ndim
            self.hr_idvs_col = None
            self.nhs, self.hr_shape = 0, 0
            self.ten_shape = ten_shape
            print("Warning: no hierarchies indicator vector collection!")
        else:
            valid, nhs, ndim, hr_shape = self._check_(hrivsc)
            if ten_ndim == ndim:
                for d in range(ndim):
                    valid &= ten_shape[d] >= hr_shape[0][d]
                if not valid:
                    print("Error: input data is invalid!")
                    exit()
                else:
                    self.ten_ndim = ndim
                    self.hr_idvs_col = hrivsc
                    self.nhs, self.hr_shape = nhs, hr_shape
                    self.ten_shape = ten_shape

    def setting(self, prob_model, encode_func=None):
        if prob_model == ProbModel.BINOMIALS:
            self.block_density_prob = binomials_encode
        elif prob_model == ProbModel.GAUSSIAN:
            self.block_density_prob = gaussian_encode
        elif prob_model == ProbModel.POISSION:
            self.block_density_prob = poisson_encode
        else:
            if encode_func is not None:
                self.block_density_prob = encode_func
            else:
                print('Error: please specify the encoding model')

    def _check_(self, hr_idvs_col):
        valid = True
        nhs, ndim, hr_shape = 0, 0, list()
        nhs = len(hr_idvs_col)

        ndim = len(hr_idvs_col[0])
        hr_shape.append([len(hr_idvs_col[0][d]) for d in range(ndim)])

        for h in range(1, nhs):
            hsp = list()
            for d in range(ndim):
                sz = len(hr_idvs_col[h][d])
                if sz <= hr_shape[-1][d]:
                    hsp.append(sz)
            if valid:
                hr_shape.append(hsp)
            else:
                break

        return valid, nhs, ndim, hr_shape

    def _block_entity_(self, hidvs, exc_pos2val=None):
        block_pos2val = dict()

        for pos, val in self.ten_sub2val.items():
            rec_valid = True
            for d in range(self.ten_ndim):
                rec_valid &= pos[d] in hidvs[d]
            if rec_valid and (not (exc_pos2val is not None and pos in exc_pos2val)):
                block_pos2val[pos] = val
        return block_pos2val

    def _encode_hridvsc_(self):
        codes = 0

        for k in range(self.nhs):
            for d in range(self.ten_ndim):
                nx0 = self.ten_shape[d] if k == 0 else self.hr_shape[k - 1][d]
                nx1 = self.hr_shape[k][d]
                p_one = nx1 * 1.0 / nx0
                entropy = - (p_one * log_2(p_one) + (1 - p_one) * log_2(1 - p_one))
                codes += nx0 * entropy + log_s(nx1)

        return codes

    def _encode_blocks_(self):
        codes = 0
        exc_posval, exc_shape = None, None  # exclude block var.
        pos2val, block_vol = None, -1

        acc_pos2val = dict()
        for h in range(self.nhs - 1, -1, -1):
            pos2val = self._block_entity_(self.hr_idvs_col[h], exc_posval)
            # mass = sum(list(pos2val.values()))
            block_vol = np.product(self.hr_shape[h])
            rem_vol = block_vol
            if exc_posval is not None:  # with exclude entities
                rem_vol -= np.prod(exc_shape)
            # density = mass * 1.0 / rem_vol
            codes += self.block_density_prob(list(pos2val.values()), rem_vol)
            # update exclude blocks
            exc_shape = block_vol.copy()
            exc_posval = pos2val.copy()
            acc_pos2val.update(pos2val)

        return codes, acc_pos2val, block_vol

    def _encode_remain_(self, maxblock_pos2val=None, maxblock_vol=-1):
        codes = 0
        remains = dict()
        if maxblock_pos2val is not None:
            for p, v in self.ten_sub2val.items():
                if p not in maxblock_pos2val:
                    remains[p] = v
        else:
            remains = self.ten_sub2val

        rem_vol = np.prod(np.asarray(self.ten_shape, float)) - maxblock_vol
        val2cnt = Counter(list(remains.values()))
        val2cnt[0] = rem_vol - len(remains)

        # entropy encode
        for val, cnt in val2cnt.items():
            codes += - cnt * log_2(cnt * 1.0 / rem_vol)
            codes += log_s(val)

        return codes

    def measure(self):
        costC = 0.0

        costC += log_s(self.nhs)
        maxblock_p2v, maxblock_vol = None, -1
        if self.nhs > 0:
            costC += self._encode_hridvsc_()
            hblk_cdlen, maxblock_p2v, maxblock_vol = self._encode_blocks_()
            costC += hblk_cdlen

        costC += self._encode_remain_(maxblock_p2v, maxblock_vol)

        return costC
