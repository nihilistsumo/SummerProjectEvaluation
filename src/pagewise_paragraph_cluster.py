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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.stats import ttest_rel
import combine_parapair_scores
import sys


def convert_qrels_to_labels(hier_qrels_file):
    hq_qrels = dict()
    with open(hier_qrels_file, 'r') as hq:
        for l in hq:
            sec = l.split(' ')[0]
            page = sec.split('/')[0]
            para = l.split(' ')[2]
            if page in hq_qrels.keys():
                hq_qrels[page][para] = sec
            else:
                hq_qrels[page] = {para: sec}
    for page in hq_qrels.keys():
        unique_sec = list(set(hq_qrels[page].values()))
        for para in hq_qrels[page].keys():
            hq_qrels[page][para] = unique_sec.index(hq_qrels[page][para])
    return hq_qrels


def compute_pagewise_ari(true_page_para_labels, cand_page_para_labels, print_pagewise, page_minmax):
    pagewise_ari_score = dict()
    minpage = ''
    maxpage = ''
    minscore = 1.0
    maxscore = 0.0
    for page in cand_page_para_labels.keys():
        true_labels = []
        cand_labels = []
        for para in cand_page_para_labels[page].keys():
            true_labels.append(true_page_para_labels[page][para])
            cand_labels.append(cand_page_para_labels[page][para])
        pagewise_ari_score[page] = adjusted_rand_score(true_labels, cand_labels)
        if print_pagewise:
            print(page+': %.4f' % (pagewise_ari_score[page]))
        if page_minmax:
            if pagewise_ari_score[page] < minscore:
                minscore = pagewise_ari_score[page]
                minpage = page
            if pagewise_ari_score[page] > maxscore:
                maxscore = pagewise_ari_score[page]
                maxpage = page
    if page_minmax:
        print('Worst page: '+minpage+', ARI: '+str(minscore))
        print('Best page: '+maxpage+', ARI: '+str(maxscore))
    return pagewise_ari_score


def cluster_paras(paras, normalized_paired_dist, splitter):
    dist_matrix = []
    for i in range(len(paras)):
        dist = []
        for j in range(len(paras)):
            if i == j:
                dist.append(0.0)
            else:
                if paras[i] + splitter + paras[j] in normalized_paired_dist.keys():
                    dist.append(1 - normalized_paired_dist[paras[i] + splitter + paras[j]])
                else:
                    dist.append(1 - normalized_paired_dist[paras[j] + splitter + paras[i]])
        dist_matrix.append(dist)
    return np.array(dist_matrix)


def pagewise_cluster(parapair_dict, norm_pair_dist, true_clusters, num_c=5, link='average'):
    page_para_labels = dict()
    splitter = '_'
    for page in parapair_dict.keys():
        parapairs = parapair_dict[page]['parapairs']
        if len(parapairs) < 1:
            continue
        labels = parapair_dict[page]['labels']
        paras = []
        para_labels = dict()
        for pp in parapairs:
            if '#' in pp:
                p1 = pp.split('#')[0]
                p2 = pp.split('#')[1]
                splitter = '#'
            else:
                p1 = pp.split('_')[0]
                p2 = pp.split('_')[1]
            if p1 not in paras:
                paras.append(p1)
            if p2 not in paras:
                paras.append(p2)
        dist_mat = cluster_paras(paras, norm_pair_dist, splitter)
        if len(dist_mat) == 0:
            print("See")
        if num_c == -1:
            true_num_cluster = max(true_clusters[page].values()) + 1
            if len(paras) < true_num_cluster:
                #print(page+' has less paras than true num clusters')
                true_num_cluster = len(paras)
            cl = AgglomerativeClustering(n_clusters=true_num_cluster, affinity='precomputed', linkage=link)
        else:
            cl = AgglomerativeClustering(n_clusters=num_c, affinity='precomputed', linkage=link)
        # cl = OPTICS(min_samples=4, metric='precomputed')
        # cl = DBSCAN(eps=0.001, min_samples=2, metric='precomputed')
        cl_labels = cl.fit_predict(dist_mat)
        # print(page + ": Max label " + str(max(cl_labels)))
        for i in range(len(paras)):
            para_labels[paras[i]] = cl_labels[i]
        page_para_labels[page] = para_labels
    return page_para_labels, splitter

def get_para_emb_dict(embid_file, embvecs_file):
    ids = np.load(embid_file)
    vecs = np.load(embvecs_file)
    para_embvec_dict = {}
    for i in range(len(ids)):
        para_embvec_dict[ids[i]] = vecs[i]
    return para_embvec_dict

def pagewise_kmeans(parapair_dict, para_emb_dict, num_c=5):
    page_para_labels = dict()
    splitter = '_'
    for page in parapair_dict.keys():
        parapairs = parapair_dict[page]['parapairs']
        if len(parapairs) < 1:
            continue
        labels = parapair_dict[page]['labels']
        paras = []
        para_labels = dict()
        para_embs = []
        for pp in parapairs:
            if '#' in pp:
                p1 = pp.split('#')[0]
                p2 = pp.split('#')[1]
                splitter = '#'
            else:
                p1 = pp.split('_')[0]
                p2 = pp.split('_')[1]
            if p1 not in paras:
                paras.append(p1)
            if p2 not in paras:
                paras.append(p2)
        for p in paras:
            para_embs.append(para_emb_dict[p])
        para_embs = np.array(para_embs)
        cl = KMeans(n_clusters=num_c, random_state=0)
        cl_labels = cl.fit_predict(para_embs)
        # print(page + ": Max label " + str(max(cl_labels)))
        for i in range(len(paras)):
            para_labels[paras[i]] = cl_labels[i]
        page_para_labels[page] = para_labels
    return page_para_labels, splitter


def main():
    parser = argparse.ArgumentParser(description='Cluster pagewise paras based on parapair score file')
    parser.add_argument('-pp', '--parapair', help='Path to para pair file')
    parser.add_argument('-hq', '--hier_qrels', help='Path to hierarchical qrels file')
    parser.add_argument('-pps', '--parapair_scores', nargs='+', help='Path to parapair score files')
    parser.add_argument('-n', '--num_cluster', type=int, help='Number of clusters for each article')
    parser.add_argument('-l', '--linkage', help='Type of linkage (complete/average/single)')
    parser.add_argument('-vi', '--pagewise', action='store_true', help='Print pagewise scores')
    parser.add_argument('-pmm', '--page_minmax', action='store_true', help='Print min and max performed pages for each method')
    args = vars(parser.parse_args())
    parapair_file = args['parapair']
    hq_file = args['hier_qrels']
    pp_score_files = args['parapair_scores']
    num_cluster = args['num_cluster']
    print_pagewise = args['pagewise']
    page_minmax = args['page_minmax']
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    methods = []
    true_page_para_labels = convert_qrels_to_labels(hq_file)
    pages = []
    for page in parapair.keys():
        if len(parapair[page]['parapairs']) > 0:
            pages.append(page)
    pages.sort()
    print("Method\tMean_ARI\tstderr")
    for pp_score_file in pp_score_files:
        method = pp_score_file.split('/')[len(pp_score_file.split('/')) - 1][:-5]
        methods.append(method)
        with open(pp_score_file, 'r') as pps:
            parapair_score = json.load(pps)

        combine_parapair_scores.minmax_normalize_ppscore_dict(parapair_score)
        link = args['linkage']
        page_para_labels, splitter = pagewise_cluster(parapair, parapair_score, true_page_para_labels, num_cluster, link)

        pagewise_ari = compute_pagewise_ari(true_page_para_labels, page_para_labels, print_pagewise, page_minmax)

        # for p in pagewise_ari.keys():
            # print(p + '\t\t%.4f' % pagewise_ari[p])
        print(method+"\t%.4f\t%.4f" % (np.mean(list(pagewise_ari.values())),
                                                np.std(list(pagewise_ari.values())) / np.sqrt(
                                                    len(list(pagewise_ari.values())))))

if __name__ == '__main__':
    main()