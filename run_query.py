#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#################################################################################
# #  CatchCore: Catching Hierarchical Dense Sub-Tensor
#  Author: wenchieh
#
#  Project: catchcore
#      run_query.py
#      Version:  2.0
#      Date: Dec. 10, 2020
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/10/2020>
#
# -------------------------------------------------------------------------------
# CatchCore Algorithm interface
#
# example:
#   python run_query.py ./example.tensor '' ./output/ 3 -1 '{0: [1], 1:[1]}' 2 3e-4 20 10 1e-6 200 ','
#
#################################################################################

__author__ = 'wenchieh'


# system
import time
import yaml
import argparse

# project
from src.hierten import HierTen
from src.toolz import initialize_tailortens
from src.utils.ioutils import load_cpd_result, save_hierten_indicators


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Dense subtensors detection for Item-Query", 
                                usage="python run_query.py infn_ten infn_cps outs dim valcol queries hs p cons etas eps maxiters sep")
    parser.add_argument("infn_ten", help="input tensor path", type=str)
    parser.add_argument("infn_cps", help="input Tensor-CP decomposition result [.mat]", type=str, default=None)
    parser.add_argument("outs", help="result output path", type=str)
    parser.add_argument("dim", help="feature dimensions", type=int)
    parser.add_argument("valcol", help="the column of 'measurement' in the input tensor", type=int)
    parser.add_argument("queries", help="the query items (dict format: dim2items)", type=yaml.safe_load)
    parser.add_argument("hs",   help="the expected number of hierarchies", type=int, default=2)
    parser.add_argument("p",    help="the penalty for missing entities", type=float, default=1e-3)
    parser.add_argument("cons", help="the Lagrange parameter for constraints of optimization func.", type=float, default=5)
    parser.add_argument("etas", help="the density ratio for two adjacent hierarchies", type=float, default=5)
    parser.add_argument("eps",  help="the convergence parameter", type=float, default=1e-6)
    parser.add_argument("maxiters",  help="the maximum number of iterations", type=int, default=100)
    parser.add_argument("sep",  help="separator of input tensor", type=str, default=',')
    args = parser.parse_args()

    ins, ins_cps, sep, outs = args.infn_ten, args.infn_cps, args.sep, args.outs
    hs, p, cons, etas, eps = args.hs, args.p, args.cons, args.etas, args.eps
    dims, val_col, max_iters = args.dim, args.valcol, args.maxiters
    queries = args.queries

    top_k = 1
    outfn = "query_hiertensor.out"
    print("ins: {}, outs: {}".format(ins, outs))
    print("hs:{}, p:{}, cons:{}, etas:{}, eps:{}".format(hs, p, cons, etas, eps))
    if ins_cps is not None:
        print("cpd:{}, top-k:{}".format(ins_cps, top_k))

    starts = time.time()
    tten, label = initialize_tailortens(ins, dims, -1, sep=sep, usecols=range(max([val_col, dims])))
    print("load data @ {} s".format(time.time() - starts))
    tten.info()
        
    selector = tten.selectormap(list(queries.values()), list(queries.keys()), direct=2)
    hrten = HierTen(tten.tosptensor())
    hrten.setting(hs, p, cons)
    alg_tm = time.time()
    print("total construct @ {} s\n".format(alg_tm - starts))

    xhs_init = None
    if ins_cps:
        print("load cps result:")
        xhs_init = load_cpd_result(ins_cps, top_k)
    
    dim2seed = dict(zip(list(queries.keys()), selector))
    xhs = hrten.queryhiers(dim2seed, xhs_init, max_iters=max_iters, eps=eps, convtol=1e-6)
    run_tm = time.time() - alg_tm
    print("\n Algorithm run @ {} s \n".format(run_tm))

    vhs, hidx, hnnzs, hshapes, hdens = hrten.hierarchy_indicator(xhs)
    print("detect index run @ {} s".format(time.time() - alg_tm))
    hrten.dump()
    print("Hierarchies density: ", hdens)

    if len(hidx) > 0:
        hr_idx = list()
        for h in range(len(hidx)):
            hr_idx.append(tten.selectormap(hidx[h], range(dims)))
        save_hierten_indicators(outs + outfn, hr_idx)

    print("done!")
