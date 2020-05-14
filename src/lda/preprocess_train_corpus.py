import argparse
from collections import defaultdict
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords

def preprocess(para_text_dict, out_token, out_corpus):
    paraids = list(para_text_dict.keys())
    raw_docs = [para_text_dict[k].lower() for k in paraids]
    print('Text lowered')
    pre_docs = [remove_stopwords(doc) for doc in raw_docs]
    print('Removed stopwords')
    c = 0
    print('Calculating frequency\n')
    frequency = defaultdict(int)
    for d in pre_docs:
        for t in d:
            frequency[t] += 1
        c += 1
        if c%1000 == 0:
            print('\rDocument processed: '+str(c), end='')
    print('\nDone')
    c = 0
    texts = []
    for doc in pre_docs:
        d = []
        for t in doc:
            if frequency[t] > 1:
                d.append(t)
        texts.append(d)
        c += 1
        if c % 1000 == 0:
            print('\rDocument processed: ' + str(c), end='')
    #texts = [[t for t in doc if frequency[t] > 1] for doc in pre_docs]
    print('\nDone')
    token_dict = corpora.Dictionary(texts)
    corpus = [token_dict.doc2bow(text) for text in texts]
    token_dict.save(out_token)
    corpora.MmCorpus.serialize(out_corpus, corpus)
    return texts, paraids

def main():
    parser = argparse.ArgumentParser(description='Preprocess training corpus')
    parser.add_argument('-pt', '--train_text', help='Path to train paratext tsv')
    parser.add_argument('-ot', '--out_token', help='Path to output token file')
    parser.add_argument('-oc', '--out_corpus', help='Path to preprocessed corpus')
    args = vars(parser.parse_args())
    paratext_file = args['train_text']
    out_token_file = args['out_token']
    out_corpus_file = args['out_corpus']
    pt_dict = {}
    with open(paratext_file, 'r') as pt:
        for l in pt:
            pt_dict[l.split('\t')[0]] = l.split('\t')[1]
    print('Paratext read, '+str(len(pt_dict))+' documents')
    preprocess(pt_dict, out_token_file, out_corpus_file)

if __name__ == '__main__':
    main()
