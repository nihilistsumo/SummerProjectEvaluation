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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.stats import ttest_rel
from pagewise_paragraph_cluster import convert_qrels_to_labels, pagewise_cluster
from combine_parapair_scores import minmax_normalize_ppscore_dict
from eval_pagewise_triple_ppscore import get_accuracy_triples
from parapair_score_evaluation import normalize_parapair_scores, read_true_parapair_dict
from pagewise_auc_comparison import calculate_auc

def compute_pagewise_ari(true_page_para_labels, cand_page_para_labels, pagelist):
    ari_score_list = []
    for i in range(len(pagelist)):
        page = pagelist[i]
        true_labels = []
        cand_labels = []
        for para in cand_page_para_labels[page].keys():
            true_labels.append(true_page_para_labels[page][para])
            cand_labels.append(cand_page_para_labels[page][para])
        ari_score_list.append(adjusted_rand_score(true_labels, cand_labels))
    return ari_score_list

def sigtest(anchor_ari_scores, scores, anchor_method, method):
    print(anchor_method+'\t\t'+method)
    #for i in range(len(anchor_ari_scores)):
    #    print('%.4f\t\t%.4f' % (anchor_ari_scores[i], scores[i]))
    print("\nMethod\t\tmean")
    print(anchor_method+'\t\t%.4f' % np.mean(anchor_ari_scores))
    print(method + '\t\t%.4f' % np.mean(scores))
    print("\nMethod1\t\tMethod2\t\tttest value\t\tp value")
    t_test = ttest_rel(anchor_ari_scores, scores)
    print(anchor_method + '\t\t' + method + '\t\t%.4f\t\t%.4f' % (t_test[0], t_test[1]))

def cluster_sigtest(parapair_file, anchor_file, pp_score_file, hq_file, num_cluster, link):
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    anchor_method = anchor_file.split('/')[len(anchor_file.split('/')) - 1][:-5]
    with open(anchor_file, 'r') as ap:
        anchor_scores = json.load(ap)
        minmax_normalize_ppscore_dict(anchor_scores)
    method = pp_score_file.split('/')[len(pp_score_file.split('/')) - 1][:-5]
    with open(pp_score_file, 'r') as pps:
        parapair_scores = json.load(pps)
        minmax_normalize_ppscore_dict(parapair_scores)
    pages = []
    for page in parapair.keys():
        if len(parapair[page]['parapairs']) > 0:
            pages.append(page)

    true_page_para_labels = convert_qrels_to_labels(hq_file)
    page_para_labels_anchor = pagewise_cluster(parapair, anchor_scores, num_cluster, link)
    anchor_ari = compute_pagewise_ari(true_page_para_labels, page_para_labels_anchor, pages)

    page_para_labels = pagewise_cluster(parapair, parapair_scores, num_cluster, link)
    other_ari = compute_pagewise_ari(true_page_para_labels, page_para_labels, pages)

    sigtest(anchor_ari, other_ari, anchor_method, method)

def triple_sigtest(parapair_file, anchor_file, pp_score_file, triple_file):
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    with open(triple_file, 'r') as trp:
        triples = json.load(trp)
    anchor_method = anchor_file.split('/')[len(anchor_file.split('/')) - 1][:-5]
    with open(anchor_file, 'r') as a:
        anchor_scores = json.load(a)
    method = pp_score_file.split('/')[len(pp_score_file.split('/')) - 1][:-5]
    with open(pp_score_file, 'r') as p:
        parapair_scores = json.load(p)
    pages = []
    for page in parapair.keys():
        if len(parapair[page]['parapairs']) > 0:
            pages.append(page)
    anchor_acc_dict = get_accuracy_triples(anchor_scores, triples)
    other_acc_dict = get_accuracy_triples(parapair_scores, triples)
    anchor_acc = []
    other_acc = []
    for p in pages:
        anchor_acc.append(anchor_acc_dict[p]['acc'])
        other_acc.append(other_acc_dict[p]['acc'])
    sigtest(anchor_acc, other_acc, anchor_method, method)

def auc_sigtest(parapair_file, anchor_file, pp_score_file):
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    anchor_method = anchor_file.split('/')[len(anchor_file.split('/')) - 1][:-5]
    with open(anchor_file, 'r') as a:
        anchor_scores = json.load(a)
    anchor_score_dict = normalize_parapair_scores(anchor_scores, 'minmax')
    method = pp_score_file.split('/')[len(pp_score_file.split('/')) - 1][:-5]
    with open(pp_score_file, 'r') as p:
        parapair_scores = json.load(p)
    parapair_score_dict = normalize_parapair_scores(parapair_scores, 'minmax')
    true_parapair_dict = read_true_parapair_dict(parapair)
    pages = []
    for page in parapair.keys():
        if len(parapair[page]['parapairs']) > 0:
            pages.append(page)
    anchor_auc = []
    other_auc = []
    for page in pages:
        _, _, auc_score_a = calculate_auc(true_parapair_dict, anchor_score_dict, page, parapair)
        anchor_auc.append(auc_score_a)
        _, _, auc_score_b = calculate_auc(true_parapair_dict, parapair_score_dict, page, parapair)
        other_auc.append(auc_score_b)
    sigtest(anchor_auc, other_auc, anchor_method, method)


def main():
    parser = argparse.ArgumentParser(description='Cluster pagewise paras based on parapair score file')
    parser.add_argument('-pp', '--parapair', help='Path to para pair file')
    parser.add_argument('-hq', '--hier_qrels', help='Path to hierarchical qrels file')
    parser.add_argument('-aps', '--anchor_scores', help='Path to anchor parapair score file')
    parser.add_argument('-pps', '--parapair_scores', help='Path to parapair score files')
    parser.add_argument('-n', '--num_cluster', type=int, help='Number of clusters for each article')
    parser.add_argument('-l', '--linkage', help='Type of linkage (complete/average/single)')
    parser.add_argument('-t', '--triples', help='Path to pagewise triples file')
    args = vars(parser.parse_args())
    parapair_file = args['parapair']
    hq_file = args['hier_qrels']
    anchor_file = args['anchor_scores']
    pp_score_file = args['parapair_scores']
    num_cluster = args['num_cluster']
    link = args['linkage']
    triple_file = args['triples']

    print("Significance test on AUC\n========================")
    auc_sigtest(parapair_file, anchor_file, pp_score_file)
    print("Significance test on Triples\n============================")
    triple_sigtest(parapair_file, anchor_file, pp_score_file, triple_file)
    print("Significance test on Clustering\n===============================")
    cluster_sigtest(parapair_file, anchor_file, pp_score_file, hq_file, num_cluster, link)

if __name__ == '__main__':
    main()
