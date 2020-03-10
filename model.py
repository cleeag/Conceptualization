import os
from os.path import join
import itertools
import pickle as pkl
from scipy.sparse import csc_matrix, bmat
from tqdm import tqdm
import numpy as np
import scipy
import csv
import time

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

from utils import read_data, read_ufet_data, co_occurence_lookup, export_result_file


class GDConcept:
    def __init__(self, tolerence, concept_num, alpha=None):
        super(GDConcept, self).__init__()
        self.tolerence = tolerence
        self.alpha = alpha
        self.concept_num = concept_num

    def _alpha_estimation(self, sum_f_list):
        print('estimating alpha')
        thresh = 5
        diff = 100
        alpha = np.full((self.concept_num, 1), 1, dtype='float')
        sum_f_arr = np.array(sum_f_list)
        while diff > thresh:
            old_alpha = np.copy(alpha)
            for t in range(self.concept_num):
                # numerator_sum, denominator_sum = 0, 0
                # for n in range(len(input_data)):
                #     tmp1 = scipy.special.digamma(old_alpha[t] + sum_f_list[n][t]) - scipy.special.digamma(old_alpha[t])
                #     tmp2 = scipy.special.digamma(sum(old_alpha + sum_f_list[n])) - scipy.special.digamma(sum(old_alpha))
                #     numerator_sum += tmp1
                #     denominator_sum += tmp2

                tmp1 = sum(
                    scipy.special.digamma(old_alpha[t] + sum_f_arr[:, t, :])
                    - scipy.special.digamma(old_alpha[t]))
                tmp2 = sum(
                    scipy.special.digamma(sum(old_alpha) + np.sum(sum_f_arr, axis=1))
                    - scipy.special.digamma(sum(old_alpha)))

                alpha[t] = old_alpha[t] * tmp1 / tmp2

            diff = abs(sum(alpha) - sum(old_alpha))
            print(diff)

        return alpha

    # get sum of feature function
    def _get_sum_of_feature(self, seq, cliques):
        # print('getting sum of features')
        sum_f = csc_matrix((self.concept_num, 1))
        for clique in cliques:
            if len(clique) == 1: continue
            print(clique)
            E_k = seq[clique[1], :]
            E_k[E_k > 0] = 1
            tmp0 = np.prod(E_k.toarray(), axis=0)
            tmp1 = csc_matrix(tmp0)
            tmp2 = seq[clique[1], :].sum(axis=0)
            sum_f += tmp1.multiply(tmp2).reshape(-1, 1)

        # one vertex clique
        for i in range(seq.shape[0]):
            E_k = seq[i, :]
            E_k[E_k > 0] = 1
            sum_f += E_k.multiply(seq[i, :]).reshape(-1, 1)

        return sum_f

    # get C_opt
    def _inference_C(self, sum_f):
        sum_f = sum_f.toarray()
        tmp = self.alpha - 1
        c_opt = (tmp + sum_f) / sum(tmp + sum_f)

        return c_opt

    # get cliques in sequence
    def _clique_detection(self, seq):
        G = nx.Graph()
        G.add_nodes_from(range(seq.shape[0]))
        for comb in itertools.combinations(range(seq.shape[0]), 2):
            cos = cosine_similarity(seq[comb[0]].reshape(1, -1), seq[comb[1]].reshape(1, -1))
            if cos > self.tolerence:
                G.add_edge(comb, 1)
        # print('nx finding cliques')
        cliques = nx.find_cliques(G)

        return cliques

    def inference(self, input_data, result_path):
        clique_list = []
        sum_f_list = []
        print('detecting cliques...')
        for idx, seq in enumerate(tqdm(input_data)):
            seq_cliques = self._clique_detection(seq)
            sum_f = self._get_sum_of_feature(seq, seq_cliques)

            clique_list.append(seq_cliques)
            sum_f_list.append(sum_f)

        if self.alpha is None:
            self.alpha = self._alpha_estimation(sum_f_list)

        print('inferencing C')
        C = []
        for idx, seq in enumerate(tqdm(input_data)):
            c_opt = self._inference_C(sum_f_list[idx])
            C.append(csc_matrix(c_opt))
        # C = bmat(C, format='csc')
        pkl.dump(C, file=open(result_path, 'wb'))
        return C


def run():
    tolerence = 0.1
    concept_num = 2359855
    instance_num = 6215859
    # input_data = [np.random.randint(20, size=(np.random.randint(3, 15), concept_num)) for _ in range(20)]
    alpha = np.full((concept_num, 1), 1)
    # alpha = csc_matrix((concept_num, 1))
    # alpha = None
    # test_data = '/home/data/cleeag/conceptualization/test_data.txt'
    test_data = '/home/data/cleeag/ufet_crowd/dev.json'
    raw_file_dir_path = '/home/data/cleeag/conceptualization/short-text-conceptualization'
    co_matrix_path = '/home/data/cleeag/conceptualization/co_matrix.pkl'
    inst2idx_dict_path = '/home/data/cleeag/conceptualization/inst2idx_dict.pkl'
    idx2concept_dict_path = '/home/data/cleeag/conceptualization/idx2concept_dict.pkl'

    annot_ids_path = '/home/data/cleeag/conceptualization/result/annot_id.pkl'
    C_result_path = '/home/data/cleeag/conceptualization/result/C.pkl'
    result_path = '/home/data/cleeag/conceptualization/result/result.txt'

    tic = time.time()
    # raw_data = read_data(test_data)
    json_data, raw_data = read_ufet_data(test_data)
    input_data, sent_data, term_data, annot_ids, idx2concept_dict = co_occurence_lookup(raw_data, concept_num,
                                                                                        instance_num, annot_ids_path,
                                                                                        raw_file_dir_path=raw_file_dir_path,
                                                                                        co_matrix_path=co_matrix_path,
                                                                                        inst2idx_dict_path=inst2idx_dict_path,
                                                                                        idx2concept_dict_path=idx2concept_dict_path)
    model = GDConcept(tolerence, concept_num, alpha=alpha)
    C = model.inference(input_data, C_result_path)
    export_result_file(sent_data, term_data, annot_ids, idx2concept_dict, C_result_path, result_path)
    toc = time.time()

    print(f'total run-time: {toc - tic:.1f}s')


if __name__ == '__main__':
    run()
