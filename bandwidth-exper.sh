#!/bin/bash

date
format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/GBoost.py -n normalization-1 > /workspace/log/"$format_date"-bandwidth-exper.log
format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/GBoost.py -n normalization-small > /workspace/log/"$format_date"-bandwidth-exper.log
date
