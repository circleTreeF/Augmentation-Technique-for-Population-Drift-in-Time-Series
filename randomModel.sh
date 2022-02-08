#!/bin/bash

for var in 21
do
  time=$(date +"%Y.%m.%d-%H-%M-%S")
  /opt/conda/bin/python GBoost.py > log/"$time"".params.GBoost.exper.""$var"".log"
  sleep 5
done
