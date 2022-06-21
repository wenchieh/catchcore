#!/usr/bin/env bash

CATCHCORE_RUN=run_catchcore.py
MDLMETRIC_RUN=run_hiertenmdl.py
QUERY_RUN=run_query.py

mkdir output
echo [DEMO] 'running ...'   # --p=3e-4
python $CATCHCORE_RUN ./example.tensor ./output/ 3 --valcol=-1 --hs=10 --p=1e-3 --cons=20 --etas=10 --eps=1e-6 --sep=','
echo [DEMO] 'done!'

echo
echo [DEMO MDL] 'tensor summary'
python $MDLMETRIC_RUN  ./example.tensor ./output/hierways.out 3 -1 binomials --sep=','
echo [DEMO MDL] 'done!'
echo 'finish'

echo
echo [DEMO] 'item query'
python $QUERY_RUN ./example.tensor '' ./output/ 3 -1 '{0: [1], 1:[1]}' 2 3e-4 20 10 1e-6 200 ','
echo [DEMO] 'done!'