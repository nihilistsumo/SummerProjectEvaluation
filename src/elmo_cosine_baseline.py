import argparse, json
import numpy as np

def cosine_sim(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos

def elmo_cosine_scores(parapair, elmo_embed):
    pair_scores = dict()
    for page in parapair.keys():
        pairs = parapair[page]['parapairs']
        print(page)
        for pair in pairs:
            p1vec = elmo_embed[()][pair.split('_')[0]]
            p2vec = elmo_embed[()][pair.split('_')[1]]
            pair_scores[pair] = float(cosine_sim(p1vec, p2vec))
    return pair_scores

def main():
    parser = argparse.ArgumentParser(description='Calculate parapair scores using elmo vec cosine sim')
    parser.add_argument('-pp', '--parapair', help='Path to para pair file')
    parser.add_argument('-el', '--elmo_embed', help='Path to elmo embed vectors')
    parser.add_argument('-o', '--out', help='Path to output file')
    args = vars(parser.parse_args())
    parapair_file = args['parapair']
    elmo_file = args['elmo_embed']
    outfile = args['out']
    with open(parapair_file, 'r') as pp:
        parapair = json.load(pp)
    elmo_embed = np.load(elmo_file, allow_pickle=True)
    parapair_scores = elmo_cosine_scores(parapair, elmo_embed)
    with open(outfile, 'w') as out:
        json.dump(parapair_scores, out)

if __name__ == '__main__':
    main()