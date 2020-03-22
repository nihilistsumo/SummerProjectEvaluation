import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics, random, math
from scipy import stats
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, StandardScaler
from scipy.stats import pearsonr, PearsonRConstantInputWarning
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import warnings

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

def calculate_pearsonr(true_parapair_dict, parapair_score_dict, page, parapair_data):
    ytrue = []
    yhat = []
    scaler = StandardScaler()
    pairs = parapair_data[page]['parapairs']
    for pp in pairs:
        ytrue.append(true_parapair_dict[pp])
        yhat.append(parapair_score_dict[pp])
    ytrue = np.array(ytrue)
    yhat = np.array(yhat)
    ytrue = scaler.fit_transform(ytrue.reshape(-1, 1)).flatten()
    yhat = scaler.fit_transform(yhat.reshape(-1, 1)).flatten()
    ps = pearsonr(ytrue, yhat)[0]
    return ps

def main():
    parser = argparse.ArgumentParser(description="Calculate regression performance measures of parapair score files")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score_files", required=True, nargs='+', help="Paths to parapair score files as list")
    parser.add_argument("-m", "--method_names", nargs='+', help="List of method names in the same order of pp score files")
    parser.add_argument("-n", "--normalization", help="Type of normalization to be used (minmax / zscore / no)")
    parser.add_argument("-me", "--metric", help="Metric to be used to measure regression performance (mse / pear)")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_files = args["parapair_score_files"]
    method_names = args["method_names"]
    norm = args["normalization"]
    metric = args["metric"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    pages = []
    for page in parapair.keys():
        if len(parapair[page]['parapairs']) > 0:
            pages.append(page)
    print("Method\t\teval score")
    nan_pages = set()
    for i in range(len(parapair_score_files)):
        parapair_score_file = parapair_score_files[i]
        with open(parapair_score_file, 'r') as pps:
            parapair_score = json.load(pps)
        parapair_score_dict = normalize_parapair_scores(parapair_score, norm)
        true_parapair_dict = read_true_parapair_dict(parapair)
        score_list = []
        for page in pages:
            if metric == 'mse':
                m = calculate_mse(true_parapair_dict, parapair_score_dict, page, parapair)
                score_list.append(m)
            elif metric == 'pear':
                m = calculate_pearsonr(true_parapair_dict, parapair_score_dict, page, parapair)
                if math.isnan(m):
                    nan_pages.add(page)
                else:
                    score_list.append(m)
        #fpr, tpr, auc_score = calculate_auc(true_parapair_dict, parapair_score_dict)
        score = np.mean(score_list)
        if method_names is None:
            method = parapair_score_file.split("/")[len(parapair_score_file.split("/")) - 1][:-5]
        else:
            method = method_names[i]
        #roc_data.append((fpr, tpr, auc_score, method))
        #print("\nAUC: "+str(calculate_auc(true_parapair_dict, parapair_score_dict)))
        print(method+"\t\t%.4f" %score)
    if len(nan_pages) > 0:
        print("Following pages caused pearsonr to return nan, most probably this means, for the following pages"
              "we have constant label for all parapairs. Hence this page is excluded from mean pearsonr calculation")
        for page in nan_pages:
            print(page)
    #draw_roc(roc_data, colors_list, title)

if __name__ == '__main__':
    main()