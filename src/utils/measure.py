#!/usr/bin/python
# -*- coding=utf-8 -*-

#  Project: catchcore
#    measure.py
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <10/23/2018>
#

__author__ = 'wenchieh'


def jaccard(pred, actual):
    intersectSize = len(set.intersection(set(pred), set(actual)))
    unionSize = len(set.union(set(pred), set(actual)))
    return intersectSize * 1.0 / unionSize


def jaccard_MD(pred, actual, dims=None):
    if dims is None:
        dims = range(len(pred))

    intersectSize, unionSize = 0.0, 0.0
    for d in dims:
        intersectSize += len(set.intersection(set(pred[d]), set(actual[d])))
        unionSize += len(set.union(set(pred[d]), set(actual[d])))
    return intersectSize * 1.0 / unionSize


def getPrecision(pred, actual):
    intersectSize = len(set.intersection(set(pred), set(actual)))
    return intersectSize * 1.0 / len(pred)


def getPrecision_MD(pred, actual, dims=None):
    if dims is None:
        dims = range(len(pred))

    intersectSize, tols = 0.0, 0.0
    for d in dims:
        intersectSize += len(set.intersection(set(pred[d]), set(actual[d])))
        tols += len(set(pred[d]))

    return intersectSize * 1.0 / tols


def getRecall(pred, actual):
    intersectSize = len(set.intersection(set(pred), set(actual)))
    return intersectSize * 1.0 / len(actual)


def getRecall_MD(pred, actual, dims=None):
    if dims is None:
        dims = range(len(pred))

    intersectSize, tols = 0.0, 0.0
    for d in dims:
        intersectSize += len(set.intersection(set(pred[d]), set(actual[d])))
        tols += len(set(actual[d]))

    return intersectSize * 1.0 / tols


def getFMeasure(pred, actual):
    prec = getPrecision(pred, actual)
    rec = getRecall(pred, actual)
    return 0 if (prec + rec == 0) else (2 * prec * rec / (prec + rec))


def getFMeasure_MD(pred, actual, dims=None):
    prec = getPrecision_MD(pred, actual, dims)
    rec = getRecall_MD(pred, actual, dims)
    return 0 if (prec + rec == 0) else (2 * prec * rec / (prec + rec))
