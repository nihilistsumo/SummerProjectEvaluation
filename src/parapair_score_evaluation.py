import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics, random
from scipy import stats
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def read_true_parapair_dict(parapair_dict):
    true_parapair_dict = dict()
    for page in parapair_dict.keys():
        pairs = parapair_dict[page]['parapairs']
        labels = parapair_dict[page]['labels']
        for i in range(len(labels)):
            true_parapair_dict[pairs[i]] = labels[i]
    return true_parapair_dict

def normalize_parapair_scores(parapair_scores, norm_method='minmax'):
    parapair_score_dict = copy.deepcopy(parapair_scores)
    if norm_method == 'minmax':
        max_score = max(list(parapair_score_dict.values()))
        min_score = min(list(parapair_score_dict.values()))
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] = (parapair_score_dict[pp] - min_score) / (max_score - min_score)
    elif norm_method == 'zscore':
        mean_score = statistics.mean(list(parapair_score_dict.values()))
        std_score = statistics.stdev(list(parapair_score_dict.values()))
        for pp in parapair_score_dict.keys():
            parapair_score_dict[pp] = (parapair_score_dict[pp] - mean_score) / std_score
        parapair_score_dict = normalize_parapair_scores(parapair_score_dict, 'minmax')
    return parapair_score_dict

def calculate_auc(true_parapair_dict, parapair_score_dict):
    ytrue = []
    yhat = []
    for pp in true_parapair_dict.keys():
        ytrue.append(true_parapair_dict[pp])
        yhat.append(parapair_score_dict[pp])
    fpr, tpr, threshold_d = metrics.roc_curve(ytrue, yhat)
    return fpr, tpr, roc_auc_score(ytrue, yhat)

def draw_roc(roc_data, colors_list, title):
    plt.title(title)
    for i in range(len(roc_data)):
        dat = roc_data[i]
        fpr = dat[0]
        tpr = dat[1]
        auc = dat[2]
        met = dat[3]
        plot_label = met+" = "+str(round(auc, 4))
        plt.plot(fpr, tpr, color=colors_list[i % len(colors_list)], label=plot_label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def main():
    # colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k']
    colors_list = list(colors.TABLEAU_COLORS.keys())
    random.shuffle(colors_list)
    parser = argparse.ArgumentParser(description="Calculate basic accuracy measures of a parapair score file")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score_files", required=True, nargs='+', help="Paths to parapair score files as list")
    parser.add_argument("-m", "--method_names", nargs='+', help="List of method names in the same order of pp score files")
    parser.add_argument("-n", "--normalization", help="Type of normalization to be used (minmax / zscore / no)")
    parser.add_argument("-t", "--plot_title", help="Title to show in ROC curve plot")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_files = args["parapair_score_files"]
    method_names = args["method_names"]
    norm = args["normalization"]
    title = args["plot_title"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    print("Method\t\tAUC score")
    roc_data = []
    for i in range(len(parapair_score_files)):
        parapair_score_file = parapair_score_files[i]
        with open(parapair_score_file, 'r') as pps:
            parapair_score = json.load(pps)
        parapair_score_dict = normalize_parapair_scores(parapair_score, norm)
        true_parapair_dict = read_true_parapair_dict(parapair)
        fpr, tpr, auc_score = calculate_auc(true_parapair_dict, parapair_score_dict)
        if method_names is None:
            method = parapair_score_file.split("/")[len(parapair_score_file.split("/")) - 1][:-5]
        else:
            method = method_names[i]
        roc_data.append((fpr, tpr, auc_score, method))
        #print("\nAUC: "+str(calculate_auc(true_parapair_dict, parapair_score_dict)))
        print(method+"\t\t%.4f" %auc_score)
    draw_roc(roc_data, colors_list, title)

if __name__ == '__main__':
    main()