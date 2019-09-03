import numpy as np
import tensorflow as tf
import json, os, argparse, copy, statistics
from scipy import stats
from scipy.stats import pearsonr
from collections import Counter
from sklearn.metrics import roc_auc_score

def rev_pp(pp):
    p1 = pp.split('_')[0]
    p2 = pp.split('_')[1]
    return p2+'_'+p1

def minmax_normalize_ppscore_dict(parapair_score_dict):
    vals = np.array(list(parapair_score_dict.values()))
    min_val = vals.min()
    max_val = vals.max()
    for pp in parapair_score_dict.keys():
        parapair_score_dict[pp] = (parapair_score_dict[pp] - min_val) / (max_val - min_val)

def get_optimized_weight(parapair_score_list, train_parapair):
    score_mat = []
    for page in train_parapair.keys():
        pairs = train_parapair[page]['parapairs']
        labels = train_parapair[page]['labels']
        for p in range(len(pairs)):
            pair = pairs[p]
            scores_for_pair = []
            for i in range(len(parapair_score_list)):
                if pair == '138f8c83335c05eca264e8d701956d2b70ab47dd_8f9961ffae3abe380ddc0a55c73479b52f66c472':
                    print('here')
                scores_for_pair.append(parapair_score_list[i][pair])
            scores_for_pair.append(labels[p])
            score_mat.append(scores_for_pair)
    score_mat = np.array(score_mat)
    np.random.shuffle(score_mat)

    ytrue = score_mat[:, score_mat.shape[1] - 1]
    score_mat = score_mat[:, :-1]

    w_init = np.array([0] * len(parapair_score_list)).reshape((len(parapair_score_list), 1))
    w = tf.Variable(w_init, trainable=True, dtype=tf.float64)
    x = tf.convert_to_tensor(score_mat)
    y = tf.convert_to_tensor(ytrue.reshape((ytrue.size, 1)))
    y_pred = tf.matmul(x, w)
    loss = tf.losses.mean_squared_error(y, y_pred)
    adam = tf.train.AdamOptimizer(learning_rate=0.01)
    a = adam.minimize(loss, var_list=w)
    with tf.Session() as sess:
        # a = adam.minimize(loss, var_list=w)
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            sess.run(a)
            if i % 1000 == 0:
                print(str(sess.run(loss)))
        optimized_fet_weight = sess.run(w).reshape((w.shape[0],))
        print("Estimated weight: " + str(optimized_fet_weight))
        weight_norm = optimized_fet_weight / sum(optimized_fet_weight)
    print("Estimated norm weights: " + str(weight_norm))
    return weight_norm

def get_combined_parapair_score(parapair_score_file_list, train_parapair_file):
    parapair_score_list = []
    for f in parapair_score_file_list:
        with open(f, 'r') as pps:
            parapair_score_dict = json.load(pps)
            minmax_normalize_ppscore_dict(parapair_score_dict)
            parapair_score_list.append(parapair_score_dict)
    with open(train_parapair_file, 'r') as tr:
        train_parapair = json.load(tr)
    weight = get_optimized_weight(parapair_score_list, train_parapair)
    comb_score = dict()
    for pp in parapair_score_list[0].keys():
        score = 0
        for i in range(len(parapair_score_list)):
            if pp in parapair_score_list[i].keys():
                score += parapair_score_list[i][pp] * weight[i]
            else:
                score += parapair_score_list[i][rev_pp(pp)] * weight[i]
        comb_score[pp] = score
    return comb_score

def main():
    parser = argparse.ArgumentParser(description="Combine parapair score files")
    parser.add_argument("-pps", "--parapair_score_files", required=True, nargs='+', help="Paths to parapair score files as list")
    parser.add_argument("-pp", "--train_parapair", required=True, help="Path to train parapair file, typically the larger split")
    parser.add_argument("-o", "--output", required=True, help="Path to combined parapair score output file")
    args = vars(parser.parse_args())
    parapair_score_files = args["parapair_score_files"]
    train_parapair_file = args["train_parapair"]
    outfile = args["output"]

    combined_parapair_score = get_combined_parapair_score(parapair_score_files, train_parapair_file)
    with open(outfile, 'w') as out:
        json.dump(combined_parapair_score, out)

if __name__ == '__main__':
    main()