#!/bin/bash

date

format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 20 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-20.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 15 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-15.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 40 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-40.log

format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 70 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-70.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 60 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-60.log

date
