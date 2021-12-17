#!/usr/bin/python
# -*- coding=utf-8 -*-

#################################################################################
# #  CatchCore: Catching Hierarchical Dense Sub-Tensor
#  Author: wenchieh
#
#  Project: catchcore
#      run_catchcore.py
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
#   python run_catchcore.py ./example.tensor ./output/ 3 -1 2 3e-4 20 10 1e-6 ','
#
#################################################################################

__author__ = 'wenchieh'

# sys
import time
import argparse

# project
from src.hierten import HierTen
from src.toolz import initialize_tailortens
from src.utils.ioutils import save_hierten_indicators

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="[CatchCore]: Catching Hierarchical Dense Sub-Tensor.",
                                     usage="python run_catchcore.py ins outs dim valcol hs p cons etas eps sep")
    parser.add_argument("--ins",  help="input tensor path", type=str, default="./example.tensor")
    parser.add_argument("--outs", help="result output path", type=str, default="./output/")
    parser.add_argument("--dim",  help="the number of feature dimensions", type=int, default=3)
    parser.add_argument("--valcol", help="the column of 'measurement' in the input tensor", type=int, default=-1)
    parser.add_argument("--hs",   help="the expected number of hierarchies", type=int, default=2)
    parser.add_argument("--p",    help="the penalty for missing entities", type=float, default=3e-4)
    parser.add_argument("--cons", help="the lagrange parameter for constraints of optimization func.", type=float, default=20)
    parser.add_argument("--etas", help="the density ratio for two adjacent hierarchies", type=float, default=10)
    parser.add_argument("--eps",  help="the convergence parameter", type=float, default=1e-6)
    parser.add_argument("--sep",  help="separator of input tensor", type=str, default=',')
    args = parser.parse_args()

    ins, sep, outs = args.ins, args.sep, args.outs
    hs, p, cons, etas, eps = args.hs, args.p, args.cons, args.etas, args.eps
    dims, valcol = args.dim, args.valcol

    outfn = 'hierways.out'
    print("\nhs:{}\npenalty: {}\netas: {}\nconstraint:{}\neps:{}\n".format(hs, p, etas, cons, eps))

    starts = time.time()
    tten, label = initialize_tailortens(
        ins, valcol, -1, sep=sep, usecols=range(max([valcol, dims])))
    print("load data @{}s".format(time.time() - starts))
    tten.info()

    hrten = HierTen(tten.tosptensor())
    hrten.setting(hs, p, cons)
    algtm = time.time()
    print("total construct @{}s\n".format(algtm - starts))

    xhs = hrten.hieroptimal(maxiter=100, eps=eps, convtol=1e-6)
    runtm = time.time() - algtm
    print("\n Algorithm run @{}s \n".format(runtm))
    vhs, hidx, hnnzs, hshapes, hdens = hrten.hierarchy_indicator(xhs)
    print("detect index run @{}s".format(time.time() - algtm))
    hrten.dump()
    print("Hierarchies density: ", hdens)

    if len(hidx) > 0:
        hridx = tten.selectormap(hidx)
        save_hierten_indicators(outs + outfn, hridx)

    print("done! " + ' @%.2f' % (runtm))
