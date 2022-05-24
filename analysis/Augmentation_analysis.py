import matplotlib.pyplot as plt
import numpy as np
import json
import os

def read_from_json(filename):
    with open(filename, 'r') as f:
        result = json.load(f)
        return result


def plot_density_hist(data):
    plt.hist(data, bins=30)  # density=False would make counts
    plt.title('Histogram of the Training Data PDF in 2006')
    plt.ylabel('Probability of Density')
    plt.xlabel('Values')
    plt.show()


def plot_train_pdf_all_year(dir):
    for file in os.listdir(dir):
        if file.endswith('.npy'):
            pdf=np.load(dir+file)
            plt.hist(pdf, bins=100)
            plt.title('Histogram of Training Data Estimated PDF in {}'.format(file[-13:-9]))
            plt.ylabel('Probability Density of Training Data in Testing Setting')
            plt.xlabel('Values')
            plt.savefig(dir+'pdf-plot/'+file[-13:-9])


def plot_train_pdf_bandwidth(dir):
    for file in os.listdir(dir):
        if file.endswith('.npy'):
            pdf=np.load(dir+file)
            plt.hist(pdf, bins=100)
            plt.title('Histogram of Training Data in 2006 Estimated PDF \n with bandwidth={}'.format(file[-10:-6]))
            plt.ylabel('Probability Density of Training Data in Testing Setting')
            plt.xlabel('Values')
            plt.savefig(dir+'pdf-plot/'+file[-8:-6]+'.png')


def latex_bandwidth_hist():
    dir = '/workspace/FYP/codespace/output/augmentation-exper/pca-exper/2009/2022-04-15-04-19-36-dim=30/pdf-plot-v3'
    count=0
    for file in sorted(os.listdir(dir)):
        bandwidth = (file[-9:-4])
        print('    \\begin{subfigure}[b]{0.48\\textwidth}\n\
                \\includegraphics[width=\\textwidth]{appendices/images/2006-bandwidth-pdf-hist/%s}\n\
                \\caption{Histogram of the Probability Density of Year 2006 Training Data Estimated with bandwidth=%s}\n\
                \\label{fig:hist-bandwidth-%s}\n\
        \\end{subfigure}' % (file, bandwidth, bandwidth))
        count+=1
        if(count%6==0):
            print('\\end{figure}\n\
\\begin{figure}[ht!]\n\
    \\ContinuedFloat')

def latex_all_years_hist(dir):
    for file in os.listdir(dir):
        print('\\begin\{subfigure\}\[b\]\{0.48\\textwidth\}\n\
            \\includegraphics[width\=\textwidth\]\{4-mainchapter/images/discussion/bandwidth-logistic-0.1-1.5-0.05.png}\
            \\caption{Comparison of Performance of Augmented Logistic Regression Classifier Given Various Bandwidth $h$ and Baseline Logistic Regression Classifier}\
            \\label{fig:logistic-bandwidth}\
        \\end{subfigure}')
