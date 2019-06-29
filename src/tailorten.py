#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: catchcore
#    Class: TailorTen
#          The tailorable tensor class to manage the tensor data
#
#    tailorten.py
#      Version:  1.0
#      Goal: Class script
#      Created by @wenchieh  on <10/23/2018>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <10/23/2018>
#


__author__ = 'wenchieh'


# third-party lib
import numpy as np
from sktensor import sptensor


class TailorTen(object):
    # input para.
    data = None   # the input tensor in dictionary format.
    ndim = None   # the dimension of input tensor
    shape = None  # input tensor shape
    nnz = 0
    vals = 0
    _dimsmin_ = None
    _dimsmap_ = None
    _invdimsmap_ = None

    def __init__(self, subs, vals, shape=None, dtype=int, accumfun=sum.__call__):
        if len(vals) <= 0:
            ValueError("the input tensor is ZERO!")

        subs = np.asarray(subs)
        ns, ndims = subs.shape
        self._dimsmin_ = np.min(subs, 0)
        self._dimsmap_ = list()
        for d in range(ndims):
            undim = np.unique(subs[:, d])
            self._dimsmap_.append(dict(zip(undim, range(len(undim)))))

        nwsubs = list()
        for k in range(ns):
            term = list()
            for d in range(ndims):
                term.append(self._dimsmap_[d][subs[k, d]])
            nwsubs.append(np.asarray(term))

        tensor = sptensor(tuple(np.asarray(nwsubs).T), np.asarray(vals), shape, dtype, accumfun=accumfun)
        self.data = dict(zip(map(tuple, np.asarray(tensor.subs).T), tensor.vals))
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.nnz = tensor.nnz()
        self.vals = np.sum(self.data.values())

    def update(self, subs, vals):
        subs = np.asarray(subs)
        ns, ndim = subs.shape
        if ndim != self.ndim:
            ValueError('input update data is invalid, the dimension is not match')

        # nwsubs = list()
        # for k in range(ns):
        # 	term = list()
        # 	for d in range(ndim):
        # 		term.append(self._dimsmap_[d][subs[k, d]])
        # 	nwsubs.append(tuple(term))

        # self.data.update(dict(zip(nwsubs, vals)))
        self.data.update(dict(zip(map(tuple, subs.T), vals)))
        print("update: nnzs ({} --> {}), vals ({} --> {})".format(self.nnz, len(self.data),
                                                                  self.vals, np.sum(self.data.values())))
        self.nnz = len(self.data)
        self.vals = np.sum(self.data.values())

    def _getinvdimsmap_(self):
        if self._invdimsmap_ is None:
            self._invdimsmap_ = [dict(zip(self._dimsmap_[dm].values(),self._dimsmap_[dm].keys()))
                                 for dm in range(self.ndim)]

    def get_entities(self, subs=None, update=False):
        if subs is None:
            ValueError("input parameter is [None]!")

        self._getinvdimsmap_()
        res = list()
        if subs is not None:
            for term in subs:
                orgterm = [self._invdimsmap_[dm][term[dm]] for dm in range(self.ndim)]
                res.append(orgterm + [self.data[tuple(term)]])
                if update:
                    del self.data[tuple(term)]
            self.nnz = len(self.data)
            self.vals = np.sum(self.data.values())

        res = np.asarray(res)
        # res[:, :-1] += self._dimsmin_
        return res

    def shave(self, sub=None):
        self.get_entities(sub, True)

    def tosptensor(self):
        return sptensor(tuple(np.asarray(self.data.keys()).T), np.asarray(self.data.values()), self.shape)

    def nnz_validsubs(self, candidates=None):
        if candidates is None:
            ValueError("input data is [None]!")
        return set(self.data.keys()).intersection(map(tuple, candidates))

    def dimension_select(self, selector):
        assert (len(selector) == self.ndim)
        res_dat = np.vstack([np.asarray(self.data.keys()).T, np.asarray(self.data.values()).T]).T
        for dm in range(self.ndim):
            res_dat = res_dat[np.isin(res_dat[:, dm], selector[dm])]
        # return map(tuple, res_dat[:, :-1]), res_dat[:, -1]
        return res_dat

    def selectormap(self, selector, direct=1):
        res = list()
        self._getinvdimsmap_()
        for h in range(len(selector)):
            hidx = list()
            for dm in range(self.ndim):
                hidx.append([self._invdimsmap_[dm][s] for s in selector[h][dm]])
            res.append(tuple(hidx))
        return res

    def info(self):
        print("dimension: {}, shape:{}, #nnz:{}".format(self.ndim, self.shape, self.nnz))
        print("initial density: {}".format(1.0 * self.nnz / np.prod(np.asarray(self.shape, float))))

    def serialize(self, outs, header=None, valtype=int, delim=',', comment='%'):
        self._getinvdimsmap_()
        with open(outs, 'w') as ofp:
            if header is not None:
                ofp.writelines(comment + header + '\n')
            for pos, v in self.data.items():
                term = [self._invdimsmap_[dm][pos[dm]] for dm in range(self.ndim)]
                ofp.writelines(delim.join(map(str, term + [valtype(v)])) + '\n')
            ofp.flush()
            ofp.close()

        print('done!')
