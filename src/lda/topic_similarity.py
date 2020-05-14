import json, math, argparse, sys
import numpy as np
from scipy.special import kl_div

EPSILON = 0.000001
def kldiv(a, b):
    score = 0
    for s in kl_div(a, b):
        if s != float("inf"):
            score += s
    return score

def sparse_jsdiv_score(v1, v2, num_topics):
    x = [0] * num_topics
    for v in v1:
        x[v[0]] = v[1]
    y = [0] * num_topics
    for v in v2:
        y[v[0]] = v[1]
    m = [(x[i]+y[i])/2 for i in range(num_topics)]
    kldiv1 = kldiv(x, m)
    kldiv2 = kldiv(y, m)
    return (kldiv1 + kldiv2)/2

def sparse_cosine_sim_score(v1, v2):
    dot_prod = 0
    for v in v1:
        id = v[0]
        for w in v2:
            if id == w[0]:
                dot_prod += v[1] * w[1]
                break
    if dot_prod < sys.float_info.min:
        return 0
    else:
        v1mod = np.sqrt(np.sum([m[1]*m[1] for m in v1]))
        v2mod = np.sqrt(np.sum([m[1] * m[1] for m in v2]))
        return dot_prod / (v1mod * v2mod)

def parapair_topic_sim(parapair, para_topic_dist, num_topics, dist_func):
    pair_scores = dict()
    for page in parapair.keys():
        pairs = parapair[page]['parapairs']
        print(page)
        for pair in pairs:
            p1vec = para_topic_dist[pair.split('_')[0]]
            p2vec = para_topic_dist[pair.split('_')[1]]
            if dist_func == 1:
                pair_scores[pair] = sparse_cosine_sim_score(p1vec, p2vec)
            elif dist_func == 2:
                pair_scores[pair] = sparse_jsdiv_score(p1vec, p2vec, num_topics)
    return pair_scores

def main():
    parser = argparse.ArgumentParser(description='Calculate parapair scores using lda topic sim')
    parser.add_argument('-pp', '--parapair', help='Path to para pair file')
    parser.add_argument('-tv', '--topic_dist', help='Path to topic dist vectors')
    parser.add_argument('-f', '--dist_func', type=int, help='1: cosine sim, 2: JS div')
    parser.add_argument('-n', '--num_topics', type=int, help='No. of topics in model')
    parser.add_argument('-o', '--out', help='Path to output file')
    args = vars(parser.parse_args())
    parapair_file = args['parapair']
    dist_file = args['topic_dist']
    dist_func = args['dist_func']
    num_t = args['num_topics']
    outfile = args['out']
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    with open(dist_file, 'r') as d:
        topic_dist = json.load(d)
    parapair_scores = parapair_topic_sim(parapair, topic_dist, num_t, dist_func)
    with open(outfile, 'w') as out:
        json.dump(parapair_scores, out)

if __name__ == '__main__':
    main()