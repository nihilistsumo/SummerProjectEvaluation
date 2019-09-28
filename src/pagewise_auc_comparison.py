import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_rel
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import parapair_score_evaluation

def calculate_auc(true_parapair_dict, parapair_score_dict, page, parapair_data):
    ytrue = []
    yhat = []
    pairs = parapair_data[page]['parapairs']
    for pp in pairs:
        ytrue.append(true_parapair_dict[pp])
        yhat.append(parapair_score_dict[pp])
    fpr, tpr, threshold_d = metrics.roc_curve(ytrue, yhat)
    return fpr, tpr, roc_auc_score(ytrue, yhat)

def calculate_sig_test(roc_dat):
    pages = list(roc_dat[random.sample(list(roc_dat.keys()), 1)[0]].keys())
    methods = list(roc_dat.keys())
    roc_mat = []
    for m in methods:
        print('\t'+m, end='')
    print('\n')
    for page in pages:
        print(page, end='')
        roc_mat_row = []
        for m in methods:
            print('\t%.4f' % roc_dat[m][page][2], end='')
            roc_mat_row.append(roc_dat[m][page][2])
        roc_mat.append(roc_mat_row)
        print('\n')
    roc_mat = np.array(roc_mat)
    print("\nMethod1\t\tMethod2\t\tttest value\t\tp value")
    for i in range(len(methods) - 1):
        for j in range(i+1, len(methods)):
            samples_a = roc_mat[:, i]
            samples_b = roc_mat[:, j]
            t_test = ttest_rel(samples_a, samples_b)
            print(methods[i]+'\t\t'+methods[j]+'\t\t%.4f\t\t%.4f' % (t_test[0], t_test[1]))


def main():
    parser = argparse.ArgumentParser(description="Compare pagewise AUC scores")
    parser.add_argument("-pp", "--parapair_file", required=True, help="Path to parapair file")
    parser.add_argument("-pps", "--parapair_score_files", required=True, nargs='+', help="Paths to parapair score files as list")
    parser.add_argument("-n", "--normalization", help="Type of normalization to be used (minmax / zscore / no)")
    args = vars(parser.parse_args())
    parapair_file = args["parapair_file"]
    parapair_score_files = args["parapair_score_files"]
    norm = args["normalization"]
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    roc_data = dict()
    true_parapair_dict = parapair_score_evaluation.read_true_parapair_dict(parapair)
    for parapair_score_file in parapair_score_files:
        with open(parapair_score_file, 'r') as pps:
            parapair_score = json.load(pps)
        parapair_score_dict = parapair_score_evaluation.normalize_parapair_scores(parapair_score, norm)
        roc_data_method = dict()
        for page in parapair.keys():
            if len(parapair[page]['parapairs']) > 0:
                fpr, tpr, auc_score = calculate_auc(true_parapair_dict, parapair_score_dict, page, parapair)
                roc_data_method[page] = (fpr, tpr, auc_score)
        method = parapair_score_file.split("/")[len(parapair_score_file.split("/")) - 1][:-5]
        roc_data[method] = roc_data_method
    calculate_sig_test(roc_data)


if __name__ == '__main__':
    main()