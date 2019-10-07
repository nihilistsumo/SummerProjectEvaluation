import argparse, json
import numpy as np

def cosine_sim(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos

def mean_glove_vec(text, glove_dir, vecsize):
    mean_vec = np.zeros(vecsize)
    c = 0
    for t in text.lower().split():
        if t in glove_dir.keys():
            mean_vec += glove_dir[t]
            c += 1
    if c > 0:
        mean_vec /= c
    return mean_vec

def glove_cosine_scores(parapair, paratext_dict, glove_dir):
    pair_scores = dict()
    vecsize = glove_dir['the'].shape[0]
    for page in parapair.keys():
        pairs = parapair[page]['parapairs']
        print(page)
        for pair in pairs:
            p1 = pair.split('_')[0]
            p2 = pair.split('_')[1]
            p1text = paratext_dict[p1]
            p2text = paratext_dict[p2]
            p1vec = mean_glove_vec(p1text, glove_dir, vecsize)
            p2vec = mean_glove_vec(p2text, glove_dir, vecsize)
            pair_scores[pair] = cosine_sim(p1vec, p2vec)
    return pair_scores

def main():
    parser = argparse.ArgumentParser(description='Calculate parapair scores using passage mean glove vector cosine sim')
    parser.add_argument('-pp', '--parapair', help='Path to para pair file')
    parser.add_argument('-pt', '--para_text', help='Path to para text file')
    parser.add_argument('-gl', '--glove', help='Path to glove vectors')
    parser.add_argument('-o', '--out', help='Path to output file')
    args = vars(parser.parse_args())
    parapair_file = args['parapair']
    paratext_file = args['para_text']
    glove_file = args['glove']
    outfile = args['out']
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    with open(paratext_file, 'r') as pt:
        paratext = json.load(pt)
    glove_dict = dict()
    with open(glove_file, 'r') as gf:
        for l in gf:
            vals = l.split()
            glove_dict[vals[0]] = np.array([float(val) for val in vals[1:]])
    pair_scores = glove_cosine_scores(parapair, paratext, glove_dict)
    with open(outfile, 'w') as out:
        json.dump(pair_scores, out)

if __name__ == '__main__':
    main()