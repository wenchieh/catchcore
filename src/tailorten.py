#!/usr/bin/python
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

    def __init__(self, subs, vals, shape=None, dtype=int, accum_fun=sum.__call__):
        if len(vals) <= 0:
            ValueError("the input tensor is ZERO!")

        subs = np.array(subs)
        ns, ndims = subs.shape
        self._dimsmin_ = np.min(subs, 0)
        self._dimsmap_ = list()
        for d in range(ndims):
            un_dim = np.unique(subs[:, d])
            self._dimsmap_.append(dict(zip(un_dim, range(len(un_dim)))))

        new_subs = list()
        for k in range(ns):
            term = list()
            for d in range(ndims):
                term.append(self._dimsmap_[d][subs[k, d]])
            new_subs.append(np.array(term))

        tensor = sptensor(tuple(np.array(new_subs).T), np.array(vals),
                          shape, dtype, accumfun=accum_fun)
        self.data = dict(zip(map(tuple, np.array(tensor.subs).T), tensor.vals))
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.nnz = len(tensor.vals)
        self.vals = np.sum(self.data.values())

        print("totals:{}, max:{}".format(np.sum(tensor.vals), np.max(tensor.vals)))

    def update(self, subs, vals):
        subs = np.array(subs)
        ns, ndim = subs.shape
        if ndim != self.ndim:
            ValueError('input update data is invalid, the dimension is not match')

        # new_subs = list()
        # for k in range(ns):
        # 	term = list()
        # 	for d in range(ndim):
        # 		term.append(self._dimsmap_[d][subs[k, d]])
        # 	new_subs.append(tuple(term))

        # self.data.update(dict(zip(new_subs, vals)))
        self.data.update(dict(zip(map(tuple, subs.T), vals)))
        print("update: nnzs ({} --> {}), vals ({} --> {})".format(self.nnz, len(self.data),
                                                                  self.vals, np.sum(self.data.values())))
        self.nnz = len(self.data)
        self.vals = np.sum(self.data.values())

    def _getinvdimsmap_(self):
        if self._invdimsmap_ is None:
            self._invdimsmap_ = [dict(zip(self._dimsmap_[dm].values(), self._dimsmap_[dm].keys()))
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

        res = np.array(res)
        return res

    def shave(self, sub=None):
        self.get_entities(sub, True)

    def tosptensor(self):
        return sptensor(tuple(np.array(list(self.data.keys())).T), np.array(list(self.data.values())), self.shape)

    def nnz_validsubs(self, candidates=None):
        if candidates is None:
            ValueError("input data is [None]!")
        return set(self.data.keys()).intersection(map(tuple, candidates))

    def dimension_select(self, selector):
        assert (len(selector) == self.ndim)
        res_dat = np.vstack([np.array(self.data.keys()).T, np.array(self.data.values()).T]).T
        for dm in range(self.ndim):
            res_dat = res_dat[np.isin(res_dat[:, dm], selector[dm])]
        # return map(tuple, res_dat[:, :-1]), res_dat[:, -1]
        return res_dat

    def selectormap(self, selectors, dims, direct=1):
        '''
        select specific entities identified by selector
        :param selector: selection subscripts
        :param direct:  program index: starting from 0 for all dimensions
                1: program index --> real index;
                2: real index --> program index;
        :return:
        '''
        res = list()
        if direct == 1:
            self._getinvdimsmap_()
        for dmidx in range(len(dims)):
            if direct == 1:
                idx = [self._invdimsmap_[dims[dmidx]][h] for h in set(selectors[dmidx])]
                res.append(idx)
                # res.append(sorted(idx))
            if direct == 2:
                idx = [self._dimsmap_[dims[dmidx]][h] for h in set(selectors[dmidx])]
                res.append(idx)
        return res

    def info(self):
        print("dimension: {}, shape:{}, #nnz:{}, ".format(self.ndim, self.shape, self.nnz))
        # print("totals:{}, max:{}".format(np.sum(self.T.vals),  np.max(self.T.vals)))
        print("initial density: {}".format(1.0 * self.nnz / np.prod(np.array(self.shape, float))))
        # print("  n_dims: {}, shape: {}, nnzs: {}, totals: {}".format(self.n_dim, self.shape,
        #                                                              len(self.T.vals), np.sum(self.T.vals)))

    def serialize(self, outs, header=None, val_type=int, delim=',', comment='%'):
        self._getinvdimsmap_()
        with open(outs, 'w') as ofp:
            if header is not None:
                ofp.writelines(comment + header + '\n')
            for pos, v in self.data.items():
                term = [self._invdimsmap_[dm][pos[dm]] for dm in range(self.ndim)]
                ofp.writelines(delim.join(map(str, term + [val_type(v)])) + '\n')
            ofp.flush()
            ofp.close()

        print('done!')
