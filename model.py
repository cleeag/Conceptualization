import os
from os.path import join
import itertools

from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn


def indicator_func(e):
    indicator = np.zeros((len(e), 1))
    indicator[np.argwhere(e > 0), 1] = 1
    return indicator


class GDConcept:
    def __init__(self, tolerence, alpha, concept_num):
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
    def _inference_C(self, alpha, seq, cliques):
        sum_f = self._get_sum_of_feature(seq, cliques)
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

    def inference(self, input_data):
        C = []
        for idx, seq in enumerate(tqdm(input_data)):
            seq_cliques = self._clique_detection(seq)
            c_opt = self._inference_C(self.alpha, seq, seq_cliques)
            C.append(c_opt)

        # print(C)
        return C



def run():
    tolerence = 0
    concept_num = 100
    input_data = [np.random.randint(20, size=(np.random.randint(3, 15), concept_num)) for _ in range(200)]
    alpha = np.full((concept_num, 1), 1)
    model = GDConcept(tolerence, alpha, concept_num)
    model.inference(input_data)


if __name__ == '__main__':
    run()
