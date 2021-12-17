#!/usr/bin/env bash

CATCHCORE_RUN=run_catchcore.py
MDLMETRIC_RUN=run_hiertenmdl.py

mkdir output
echo [DEMO] 'running ...'
python $CATCHCORE_RUN ./example.tensor ./output/ 3 -1 2 3e-4 20 10 1e-6 ','
echo [DEMO] 'done!'

echo
echo [DEMO MDL] 'tensor summary'
python $MDLMETRIC_RUN  ./example.tensor ./output/hierways.out 3 -1 binomials ','
echo [DEMO MDL] 'done!'
echo 'finish'