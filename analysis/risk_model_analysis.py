import pickle
import os
import matplotlib.pyplot as plt


def random_model_aggregate():
    dir = '/workspace/FYP/codespace/output/random/'
    roc_list = []
    x = list(range(2006, 2020))
    num = 0
    for file in os.listdir(dir):
        f = open(dir + file, 'rb')
        result = pickle.load(f)
        roc_list.append(result['all year AUC'][:14])
        plt.plot(x, result['all year AUC'][:14])
        num = num + 1
        print(file)
    plt.title("AUC of Gradient Boosting - " + str(num) + " Times")
    plt.xlabel("Year")
    plt.ylabel("AUC")
    plt.show()

def basemodel_tuning(dir):
    auc_list = []
    x = list(range(2006, 2021))
    num = 0
    for file in os.listdir(dir):
        f = open(dir + file, 'rb')
        result = pickle.load(f)
        auc_list.append(result['all year AUC'])
        plt.plot(x, result['all year AUC'])
        num = num + 1
        print(file)
        if num==3:
            break
    plt.title("AUC of Gradient Boosting - " + str(num) + " Times")
    plt.xlabel("Year")
    plt.ylabel("AUC")
    plt.show()

if __name__ == '__main__':
    random_model_aggregate()
