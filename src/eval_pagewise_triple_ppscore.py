import json, sys, argparse
import numpy as np
from scipy.stats import ttest_rel

def get_parapair_score(parapair_scores, pair):
    if pair in parapair_scores.keys():
        return parapair_scores[pair]
    else:
        p1 = pair.split('_')[0]
        p2 = pair.split('_')[1]
        if p2+'_'+p1 in parapair_scores.keys():
            return parapair_scores[p2+'_'+p1]
        else:
            print(pair+" not in parapair score dict!")
            return -1

def get_accuracy_triples(parapair_scores, pagewise_triples):
    accuracy_dict = dict()
    for page in pagewise_triples.keys():
        hit = 0
        missing_score = 0
        num = len(pagewise_triples[page])
        for t in pagewise_triples[page]:
            p1 = t[0]
            p2 = t[1]
            p3 = t[2]
            odd = t[3]
            p1p2_score = get_parapair_score(parapair_scores, p1 + '_' + p2)
            p2p3_score = get_parapair_score(parapair_scores, p2 + '_' + p3)
            p3p1_score = get_parapair_score(parapair_scores, p3 + '_' + p1)
            if p1p2_score < 0 or p2p3_score < 0 or p3p1_score < 0:
                missing_score += 1
                continue

            if p1p2_score > p2p3_score and p1p2_score > p3p1_score:
                guessed_odd = p3
            elif p2p3_score > p1p2_score and p2p3_score > p3p1_score:
                guessed_odd = p1
            else:
                guessed_odd = p2
            if guessed_odd == odd:
                hit += 1
        accuracy_dict[page] = {'hit':hit, 'num':num, 'missing_score':missing_score, 'acc':(hit/num)}
        #print(page)
    return accuracy_dict

def main():
    parser = argparse.ArgumentParser(description='Evaluate pagewise triple accuracy using parapair scores')
    parser.add_argument('-ps', '--parapair_scores', nargs='+', help='Path to parapair score files as a list')
    parser.add_argument('-t', '--page_triples', help='Path to pagewise triples file')
    args = vars(parser.parse_args())
    parapair_score_files = args['parapair_scores']
    triple_file = args['page_triples']
    parapair_score_files.sort()
    with open(triple_file, 'r') as trp:
        triples = json.load(trp)
    pages = list(triples.keys())
    pages.sort()
    method_list = []
    print("Method\t\tTriplet accuracy")
    page_acc_mat = []
    for parapair_score_file in parapair_score_files:
        with open(parapair_score_file, 'r') as ps:
            parapair_scores = json.load(ps)
        pagewise_acc = get_accuracy_triples(parapair_scores, triples)
        method = parapair_score_file.split('/')[len(parapair_score_file.split('/')) - 1]
        method_list.append(method)
        pagewise_acc_row = [pagewise_acc[pages[p]]['acc'] for p in range(len(pages))]

        page_acc_mat.append(pagewise_acc_row)
        # for p in pagewise_acc.keys():
            # print(p+' hit: '+str(pagewise_acc[p]['hit'])+', num: '+str(pagewise_acc[p]['num'])+', missing: '+
                  # str(pagewise_acc[p]['missing_score'])+', acc: '+str(pagewise_acc[p]['acc']))

        mean_acc = np.mean([pagewise_acc[p]['acc'] for p in pages])
        print(method+"\t\t%.4f" % mean_acc)
    page_acc_mat = np.array(page_acc_mat)
    np.transpose(page_acc_mat)
    print("\nMethod1\t\tMethod2\t\tttest value\t\tp value")
    for i in range(len(method_list) - 1):
        for j in range(i + 1, len(method_list)):
            samples_a = page_acc_mat[:, i]
            samples_b = page_acc_mat[:, j]
            t_test = ttest_rel(samples_a, samples_b)
            print(method_list[i] + '\t\t' + method_list[j] + '\t\t%.4f\t\t%.4f' % (t_test[0], t_test[1]))

if __name__ == '__main__':
    main()