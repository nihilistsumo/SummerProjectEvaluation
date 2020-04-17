from significance_tests import auc_sigtest, cluster_sigtest
import argparse

def all_sigtest(parapair_file, anchor_file, pps_files, hq_file, num_cluster, link):
    auc_sig_test_results = []
    cluster_sig_test_results = []
    for f in pps_files:
        auc_sig_test_results.append(auc_sigtest(parapair_file, anchor_file, f))
        cluster_sig_test_results.append(cluster_sigtest(parapair_file, anchor_file, f, hq_file, num_cluster, link))
    print('Anchor method\t\tOther method\t\tAUC ttest\t\tAUC p\t\tClust ttest\t\tClust p')
    for i in range(len(auc_sig_test_results)):
        print(auc_sig_test_results[i]['anchor']+'\t\t'+auc_sig_test_results[i]['method']+'\t\t'
              +str(auc_sig_test_results[i]['ttest'])+'\t\t'+str(auc_sig_test_results[i]['pval'])+'\t\t'
              +str(cluster_sig_test_results[i]['ttest'])+'\t\t'+str(cluster_sig_test_results[i]['pval']))

def main():
    parser = argparse.ArgumentParser(description='Sigtest on list of para scores based on anchor')
    parser.add_argument('-pp', '--parapair', help='Path to para pair file')
    parser.add_argument('-hq', '--hier_qrels', help='Path to hierarchical qrels file')
    parser.add_argument('-aps', '--anchor_scores', help='Path to anchor parapair score file')
    parser.add_argument('-pps', '--parapair_scores', nargs='+', help='Path to parapair score files')
    parser.add_argument('-n', '--num_cluster', type=int, help='Number of clusters for each article')
    parser.add_argument('-l', '--linkage', help='Type of linkage (complete/average/single)')
    args = vars(parser.parse_args())
    parapair_file = args['parapair']
    hq_file = args['hier_qrels']
    anchor_file = args['anchor_scores']
    pp_score_files = args['parapair_scores']
    num_cluster = args['num_cluster']
    link = args['linkage']

    all_sigtest(parapair_file, anchor_file, pp_score_files, hq_file, num_cluster, link)

if __name__ == '__main__':
    main()