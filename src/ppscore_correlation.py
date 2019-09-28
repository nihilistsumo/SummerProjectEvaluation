import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics
from scipy import stats
from scipy.stats import pearsonr
from collections import Counter
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import parapair_score_evaluation


def draw_corr_heatmap(method_list, corr_matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(method_list)))
    ax.set_yticks(np.arange(len(method_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(list(range(1, len(method_list)+1)))
    ax.set_yticklabels(list(range(1, len(method_list)+1)))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Pearsons correlation matrix")
    fig.tight_layout()
    plt.show()

def draw_corr_sb_hm(method_list, corr_matrix):
    df = pd.DataFrame(corr_matrix, index=list(range(1, len(method_list)+1)), columns=list(range(1, len(method_list)+1)))
    ax = sns.heatmap(df)
    ax.set_title("Pearsons correlation matrix")
    plt.show()

def calculate_pp_score_correlation(parapair_score_dicts):
    methods = list(parapair_score_dicts.keys())
    methods.sort()
    cor_matrix = []
    print('Following is the method list:')
    for i in range(len(methods)):
        print(str(i+1)+'. '+methods[i])
    print('Pearsons correlation matrix')
    for i in range(len(methods)):
        print('\t'+str(i+1), end='')
    print('\n')
    for i in range(len(methods)):
        print(str(i+1)+'\t', end='')
        corr = []
        for j in range(len(methods)):
            m1 = methods[i]
            m2 = methods[j]
            score_dict1 = parapair_score_dicts[m1]
            score_dict2 = parapair_score_dicts[m2]
            x = []
            y = []
            for p in score_dict1.keys():
                x.append(score_dict1[p])
                y.append(score_dict2[p])
            pearson_cor, _ = pearsonr(x, y)
            print('%.4f\t' % pearson_cor, end='')
            corr.append(pearson_cor)
        print('\n')
        cor_matrix.append(corr)
    return methods, cor_matrix

def main():
    parser = argparse.ArgumentParser(description="Calculate correlations between parapair score files")
    parser.add_argument("-pps", "--parapair_score_files", required=True, nargs='+', help="Paths to parapair score files as list")
    parser.add_argument("-n", "--normalization", help="Type of normalization to be used (minmax / zscore / no)")
    args = vars(parser.parse_args())
    parapair_score_files = args["parapair_score_files"]
    norm = args["normalization"]

    parapair_score_dicts = dict()
    for parapair_score_file in parapair_score_files:
        with open(parapair_score_file, 'r') as pps:
            parapair_score = json.load(pps)
        method = parapair_score_file.split('/')[len(parapair_score_file.split('/')) - 1]
        parapair_score_dicts[method] = parapair_score_evaluation.normalize_parapair_scores(parapair_score, norm)
    m, mat = calculate_pp_score_correlation(parapair_score_dicts)
    draw_corr_sb_hm(m, mat)

if __name__ == '__main__':
    main()