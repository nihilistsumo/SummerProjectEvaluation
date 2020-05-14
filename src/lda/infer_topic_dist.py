import spacy, nltk, gensim, json, argparse
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
from gensim.models import ldamodel

def preprocess(para_text_dict):
    stops = stopwords.words('english')
    paraids = list(para_text_dict.keys())
    raw_docs = [para_text_dict[k] for k in paraids]
    pre_docs = [[word for word in doc.lower().split() if word not in stops] for doc in raw_docs]
    frequency = defaultdict(int)
    for d in pre_docs:
        for t in d:
            frequency[t] += 1
    texts = [[t for t in doc if frequency[t] > 1] for doc in pre_docs]
    return texts, paraids

def infer_topics_lda(para_text_dict, stops, model, token_dict):
    topic_vector_dict = dict()
    unseen_texts, paraids = preprocess(para_text_dict)
    unseen_corpus = [token_dict.doc2bow(text) for text in unseen_texts]
    for p in range(len(paraids)):
        topic_vec = model[unseen_corpus[p]]
        topic_vector_dict[paraids[p]] = [(t[0], float(t[1])) for t in topic_vec]
    return topic_vector_dict

def main():
    parser = argparse.ArgumentParser(description='Infer topic distribution')
    parser.add_argument('-t', '--token_dict', help='Path to train token dictionary')
    parser.add_argument('-m', '--model', help='Path to LDA model file')
    parser.add_argument('-tp', '--test_paratext', help='Path to test paratext tsv file')
    parser.add_argument('-o', '--out', help='Path to para topic dist output file')
    args = vars(parser.parse_args())
    token_dict_file = args['token_dict']
    model_file = args['model']
    test_pt_file = args['test_paratext']
    pt_dict = {}
    with open(test_pt_file, 'r') as pt:
        for l in pt:
            pt_dict[l.split('\t')[0]] = l.split('\t')[1]
    outfile = args['out']
    stops = stopwords.words('english')
    d = corpora.Dictionary.load(token_dict_file)
    model = ldamodel.LdaModel.load(model_file)
    para_topic_dict = infer_topics_lda(pt_dict, stops, model, d)
    for p in para_topic_dict.keys():
        print(p+': '+str(para_topic_dict[p]))
    with open(outfile, 'w') as o:
        json.dump(para_topic_dict, o)

if __name__ == '__main__':
    main()