#!/usr/bin/python
# -*- coding=utf-8 -*-

#################################################################################
# #  CatchCore: Catching Hierarchical Dense Sub-Tensor
#  Author: wenchieh
#
#  Project: catchcore
#      run_hiertenmdl.py
#      Version:  1.0
#      Date: Oct. 30 2018
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <10/30/2018>
#
# -------------------------------------------------------------------------------
# CatchCore Algorithm interface
#
# example:
#   python run_hiertenmdl.py ./example.tensor ./output/hierways.out 3 -1 binomials ','
#
#################################################################################

__author__ = 'wenchieh'

# sys
import argparse

# project
from src.toolz import initialize_tailortens
from src.mdlmodel import MDLModel, ProbModel
from src.utils.ioutils import load_hierten_indicators


models = {
    'binomials': ProbModel.BINOMIALS,
    'poisson': ProbModel.POISSION,
    'gaussian': ProbModel.GAUSSIAN
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="[HierTenMDL]: Minimum Description Length for hierarchical dense subtensors.",
                                     usage="python run_hiertenmdl.py infn_ten infn_hts dim valcol modeltype sep")
    parser.add_argument("infn_ten", help="input tensor path", type=str)
    parser.add_argument("infn_hts", help="input hierarchical subtensor file path", type=str)
    parser.add_argument("dim", help="feature dimensions", type=int)
    parser.add_argument("valcol", help="the column of 'measurement' in the input tensor", type=int)
    parser.add_argument("modeltype", help="the model type for entity distribution [default: poisson]",
                        type=str, default="poisson", choices=['binomials', 'poisson', 'gaussian'])
    parser.add_argument("sep", help="separator of input tensor", type=str, default=' ')
    args = parser.parse_args()

    infn_ten, infn_hts = args.infn_ten, args.infn_hts
    ndim, valcol = args.dim, args.valcol
    modeltype = str(args.modeltype)
    sep = args.sep

    print("Information:")
    tten, _ = initialize_tailortens(infn_ten, valcol, -1, sep=sep)
    print("\t tensor info: ndim:{} shape:{}".format(tten.ndim, tten.shape))

    nhs, ndim, hidvs_col = load_hierten_indicators(infn_hts)
    print("\t Hierarchies info: Nhs: {}, ndim: {}".format(nhs, ndim))

    model = models[modeltype]
    print("\t Probability model: {}".format(repr(model)))
    gmdl0 = MDLModel(tten.data)
    gmdl0.setting(model)
    costC0 = gmdl0.measure()
    print("[No Hierarchical Dense Subtenor] MDL cost: {}".format(costC0))
    gmdl = MDLModel(tten.data, hidvs_col)
    gmdl.setting(model)
    costC = gmdl.measure()
    print("MDL cost: {}".format(costC))
    print("done!")
