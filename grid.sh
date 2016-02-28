#!/bin/bash

python paramScan.py 200000 10 0.01  &
python paramScan.py 200000 10 0.001 &

python paramScan.py 100000 10 0.1   &
python paramScan.py 100000 10 0.01  &
python paramScan.py 100000 10 0.001 &

python paramScan.py  50000 20 0.1   &
python paramScan.py  50000 20 0.01  &
python paramScan.py  50000 20 0.001 &
