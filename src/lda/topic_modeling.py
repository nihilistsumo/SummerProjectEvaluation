import spacy, nltk, gensim, json, argparse
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
from gensim.models import ldamodel
from nltk.corpus import stopwords

# def preprocess(para_text_dict, stops):
#     paraids = list(para_text_dict.keys())
#     raw_docs = [para_text_dict[k] for k in paraids]
#     pre_docs = [[word for word in doc.lower().split() if word not in stops] for doc in raw_docs]
#     frequency = defaultdict(int)
#     for d in pre_docs:
#         for t in d:
#             frequency[t] += 1
#     texts = [[t for t in doc if frequency[t] > 1] for doc in pre_docs]
#     return texts, paraids
#
# def save_corpus(texts, out_dict_file, out_corpus_file):
#     token_dict = corpora.Dictionary(texts)
#     corpus = [token_dict.doc2bow(text) for text in texts]
#     token_dict.save(out_dict_file)
#     corpora.MmCorpus.serialize(out_corpus_file, corpus)

def train_lda(corpus, token_dict, num_topics, update, passes):
    return ldamodel.LdaModel(corpus=corpus, id2word=token_dict, num_topics=num_topics, update_every=update, passes=passes)

def main():
    parser = argparse.ArgumentParser(description='Cluster pagewise paras based on parapair score file')
    parser.add_argument('-c', '--train_corpus', help='Path to train corpus')
    parser.add_argument('-t', '--token_dict', help='Path to train token dictionary')
    parser.add_argument('-n', '--num_topics', type=int, help='Number of topics')
    parser.add_argument('-u', '--update', type=int, help='Update freq')
    parser.add_argument('-p', '--passes', type=int, help='No. of passes')
    parser.add_argument('-op', '--out', help='Path to LDA model output file')
    args = vars(parser.parse_args())
    train_corpus_file = args['train_corpus']
    token_dict_file = args['token_dict']
    num_t = args['num_topics']
    u = args['update']
    p = args['passes']
    outfile = args['out']

    c = corpora.MmCorpus(train_corpus_file)
    d = corpora.Dictionary.load(token_dict_file)
    model = train_lda(c, d, num_t, u, p)
    model.save(outfile)

if __name__ == '__main__':
    main()
