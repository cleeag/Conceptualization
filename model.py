import os
from os.path import join
import itertools

from tqdm import tqdm
import numpy as np
import scipy
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn


def indicator_func(e):
    indicator = np.zeros((len(e), 1))
    indicator[np.argwhere(e > 0), 1] = 1
    return indicator


class GDConcept:
    def __init__(self, tolerence, concept_num, alpha=None):
        super(GDConcept, self).__init__()
        self.tolerence = tolerence
        self.alpha = alpha
        self.concept_num = concept_num

    # get sum of feature function
    def _get_sum_of_feature(self, seq, cliques):
        sum_f = np.zeros((self.concept_num, 1))
        for clique in cliques:
            if len(clique) == 1: continue
            E_k = seq[clique[1], :]
            E_k[E_k > 0] = 1
            sum_f += np.multiply(np.prod(E_k, axis=0), np.sum(E_k, axis=0)).reshape(-1, 1)

        return sum_f

    # get C_opt
    def _inference_C(self, alpha, sum_f):
        # sum_f = self._get_sum_of_feature(seq, cliques)
        c_opt = (alpha - 1 + sum_f) / sum(alpha - 1 + sum_f)

        return c_opt

    # get cliques in sequence
    def _clique_detection(self, seq):
        G = nx.Graph()
        G.add_nodes_from(range(len(seq)))
        for comb in itertools.combinations(range(len(seq)), 2):
            if cosine_similarity(seq[comb[0]].reshape(1, -1), seq[comb[1]].reshape(1, -1)) > self.tolerence:
                G.add_edge(comb, 1)
        cliques = nx.find_cliques(G)

        return cliques

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
                #
                #     numerator_sum += tmp1
                #     denominator_sum += tmp2

                tmp1 = sum(scipy.special.digamma(old_alpha[t] + sum_f_arr[:, t, :]) - scipy.special.digamma(old_alpha[t]))
                tmp2 = sum(scipy.special.digamma(sum(old_alpha) + np.sum(sum_f_arr, axis=1)) - scipy.special.digamma(sum(old_alpha)))

                alpha[t] = old_alpha[t] * tmp1 / tmp2

            diff = abs(sum(alpha) - sum(old_alpha))
            print(diff)

        return alpha



    def inference(self, input_data):
        clique_list = []
        sum_f_list = []
        for idx, seq in enumerate(tqdm(input_data)):
            seq_cliques = self._clique_detection(seq)
            sum_f = self._get_sum_of_feature(seq, seq_cliques)

            clique_list.append(seq_cliques)
            sum_f_list.append(sum_f)

        if not self.alpha:
            self.alpha = self. _alpha_estimation(sum_f_list)

        C = []
        for idx, seq in enumerate(tqdm(input_data)):
            c_opt = self._inference_C(self.alpha, sum_f_list[idx])
            C.append(c_opt)

        print(C[0])
        return C



def run():
    tolerence = 0
    concept_num = 100
    input_data = [np.random.randint(20, size=(np.random.randint(3, 15), concept_num)) for _ in range(20)]
    # alpha = np.full((concept_num, 1), 1)
    model = GDConcept(tolerence, concept_num)
    model.inference(input_data)


if __name__ == '__main__':
    run()
