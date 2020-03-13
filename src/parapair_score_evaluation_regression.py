import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics, random
from scipy import stats
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
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

# def calculate_auc(true_parapair_dict, parapair_score_dict):
#     ytrue = []
#     yhat = []
#     for pp in true_parapair_dict.keys():
#         ytrue.append(true_parapair_dict[pp])
#         yhat.append(parapair_score_dict[pp])
#     fpr, tpr, threshold_d = metrics.roc_curve(ytrue, yhat)
#     return fpr, tpr, roc_auc_score(ytrue, yhat)
def calculate_mse(true_parapair_dict, parapair_score_dict, page, parapair_data):
    ytrue = []
    yhat = []
    pairs = parapair_data[page]['parapairs']
    for pp in pairs:
        ytrue.append(true_parapair_dict[pp])
        yhat.append(parapair_score_dict[pp])
    mse = mean_squared_error(ytrue, yhat)
    return mse


def main():
    parser = argparse.ArgumentParser(description="Calculate regression performance measures of parapair score files")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score_files", required=True, nargs='+', help="Paths to parapair score files as list")
    parser.add_argument("-m", "--method_names", nargs='+', help="List of method names in the same order of pp score files")
    parser.add_argument("-n", "--normalization", help="Type of normalization to be used (minmax / zscore / no)")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_files = args["parapair_score_files"]
    method_names = args["method_names"]
    norm = args["normalization"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    pages = []
    for page in parapair.keys():
        if len(parapair[page]['parapairs']) > 0:
            pages.append(page)
    print("Method\t\tMSE score")
    for i in range(len(parapair_score_files)):
        parapair_score_file = parapair_score_files[i]
        with open(parapair_score_file, 'r') as pps:
            parapair_score = json.load(pps)
        parapair_score_dict = normalize_parapair_scores(parapair_score, norm)
        true_parapair_dict = read_true_parapair_dict(parapair)
        mse_list = []
        for page in pages:
            m = calculate_mse(true_parapair_dict, parapair_score_dict, page, parapair)
            mse_list.append(m)
        #fpr, tpr, auc_score = calculate_auc(true_parapair_dict, parapair_score_dict)
        mse_score = np.mean(mse_list)
        if method_names is None:
            method = parapair_score_file.split("/")[len(parapair_score_file.split("/")) - 1][:-5]
        else:
            method = method_names[i]
        #roc_data.append((fpr, tpr, auc_score, method))
        #print("\nAUC: "+str(calculate_auc(true_parapair_dict, parapair_score_dict)))
        print(method+"\t\t%.4f" %mse_score)
    #draw_roc(roc_data, colors_list, title)

if __name__ == '__main__':
    main()