#!/bin/bash

date
format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 30 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-30.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 25 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-25.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 35 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-35.log

format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 10 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-10.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
nohup /opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 5 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-5.log &
format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 50 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-50.log
format_date=`date +"%Y-%m-%d-%H-%M"`


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

format_date=`date +"%Y-%m-%d-%H-%M"`
/opt/conda/bin/python /workspace/FYP/codespace/PCA_ClassifierNonRandom.py -n norm-1 -d 80 > /workspace/FYP/codespace/log/"$format_date"-pca-bandwidth-norm-1-exper-80.log

date
