#!/usr/bin/python
# -*- coding=utf-8 -*-

#  Project: catchcore
#    ioutils.py
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <10/2/2018>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <10/2/2018>
#


__author__ = 'wenchieh'

# sys
import os
import sys

# third-party libs
from scipy import io
import numpy as np


def append_suffix(fn, suffix):
    fn_out, fnext_out = os.path.splitext(os.path.basename(fn))
    return fn_out + suffix + fnext_out


def read_file(fn, mode='r'):
    if '.gz' == fn[-3:]:
        fn = fn[:-3]
    if os.path.isfile(fn):
        f = open(fn, mode)
    elif os.path.isfile(fn + '.gz'):
        import gzip
        f = gzip.open(fn + '.gz', mode)
    else:
        ValueError('File: {} or its zip file dose NOT exist'.format(fn))
        sys.exit(1)
    return f


def save_simple_dictdata(sim_dict, outfn, sep=':'):
    with open(outfn, 'w') as fp:
        for k, v in sim_dict.iteritems():
            fp.writelines('{}{}{}\n'.format(k, sep, v))
        fp.close()


def load_simple_dictdata(infn, key_type=int, val_type=float, sep=':'):
    sim_dict = dict()
    with read_file(infn, 'r') as fp:
        for line in fp:
            tokens = line.strip().split(sep)
            sim_dict[key_type(tokens[0])] = val_type(tokens[1])
        fp.close()
    return sim_dict


def save_dictlist(dict_list, outfn, sep_dict=':', sep_list=','):
    with open(outfn, 'w') as fp:
        for k, ls in dict_list.iteritems():
            if type(ls) is not list:
                ValueError('The value of the data is NOT list type!.')
                break
            ls_str = sep_list.join(str(t) for t in ls)
            fp.writelines('{}{}{}\n'.format(k, sep_dict, ls_str))
        fp.close()


def load_dictlist(infn, key_type=str, val_type=str, sep_dict=':', sep_list=','):
    dict_list = dict()
    with read_file(infn, 'r') as fp:
        for line in fp:
            tokens = line.strip().split(sep_dict)
            lst = [val_type(tok) for tok in tokens[1].strip().split(sep_list)]
            dict_list[key_type(tokens[0])] = lst
        fp.close()

    return dict_list


def save_simple_list(sim_list, outfn):
    with open(outfn, 'w') as fp:
        line_str = '\n'.join(str(t) for t in sim_list)
        fp.writelines(line_str)
        fp.close()


def load_simple_list(infn, dtype=None):
    sim_list = list()
    with read_file(infn, 'r') as fp:
        for line in fp:
            t = line.strip()
            if t == '':
                continue
            if dtype is not None:
                t = dtype(t)
            sim_list.append(t)
        fp.close()

    return sim_list


def save_hierten_indicators(outfn, hier_indicator, sep_hdm2ids=":", sep_inds=","):
    nhs = len(hier_indicator)
    ndim = len(hier_indicator[0])

    with open(outfn, 'w') as fp:
        fp.writelines("% {},{}\n".format(nhs, ndim))
        shape_str = ''
        for h in range(nhs):
            hsp = 'x'.join(map(str, [len(hier_indicator[h][dm]) for dm in range(ndim)]))
            shape_str += hsp + '\t'
        fp.writelines('% {}\n'.format(shape_str))

        for h in range(nhs):
            for dm in range(ndim):
                ind_str = sep_inds.join(map(str, hier_indicator[h][dm]))
                outs = "{}_{}{}{}".format(h, dm, sep_hdm2ids, ind_str)
                fp.writelines(outs + '\n')
        fp.close()


def load_hierten_indicators(infn, sep_hdm2ids=":", sep_inds=","):
    res = list()
    with open(infn, 'r') as fp:
        line = fp.readline().strip()[1:]
        nhs, ndim = map(int, line.split(','))
        fp.readline()
        for h in range(nhs):
            hids = dict()
            for dm in range(ndim):
                line = fp.readline().strip()
                toks = line.split(sep_hdm2ids)
                _, dm = map(int, toks[0].strip().split('_'))
                inds = list(map(int, toks[1].strip().split(sep_inds)))
                hids[dm] = inds
            sort_hids = list()
            for dm in sorted(list(hids.keys())):
                sort_hids.append(np.array(hids[dm]))
            res.append(sort_hids)
        fp.close()

    return nhs, ndim, res


def load_mat(infn, var_name='data'):
    mat = dict(io.loadmat(infn, appendmat=True, squeeze_me=True))
    subs = mat[var_name]['subs'].tolist()
    vals = mat[var_name]['vals'].tolist()
    return subs, vals


def load_tensor(infn, val_col=-1, val_type=int, label_col=-1, label_type=str, sep=',', comment='%'):
    idxs, vals, labels = list(), list(), list()
    with open(infn, 'r') as fp:
        for line in fp:
            if line.startswith(comment):
                continue
            toks = line.strip().split(sep)
            if val_col > 0:
                vals.append(val_type(toks[val_col]))
            else:
                vals.append(val_type(1))

            if label_col > 0:
                labels.append(label_type(toks[label_col]))

            pt_idx = list()
            for k in range(len(toks)):
                if k != val_col and k != label_col:
                    pt_idx.append(int(toks[k]))
            idxs.append(np.array(pt_idx))

        fp.close()
    # subs = np.array(idxs) #list(np.array(idxs).transpose())
    return np.array(idxs), vals, labels


def load_cpd_result(infn, top_r):
    dat = io.loadmat(infn, squeeze_me = True)
    us = np.squeeze(dat['rkP']['u']).item()

    hxs_init = list()
    n_dim = len(us)
    for r in range(top_r):
        hxs = list()
        for dm in range(n_dim):
            xdm = us[dm][:,r]
            if xdm.sum() < 0:
                xdm *= -1
            xdm[xdm <= 0] = 1e-5
            hxs.append(xdm)
        hxs_init.append(hxs)

    return hxs_init