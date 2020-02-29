import os
from os.path import join
import itertools
import pickle as pkl
from scipy.sparse import *
from tqdm import tqdm


def read_data(path):
    data = []
    with open(path, 'r') as r:
        for i, line in enumerate(r):
            data.append(line.split())
    return data

def co_occurence_lookup(data, concept_num, instance_num, co_path='', co_matrix_path='', inst2idx_path='', inst2idx_dict_path=''):
    if not os.path.exists(co_matrix_path):
        co_matrix = lil_matrix((concept_num, instance_num))
        with open(co_path, 'r') as r:
            for i, line in enumerate(tqdm(r)):
                concept, instance, co_value = [int(x) for x in line.split()]
                co_matrix[concept, instance] = co_value
        co_matrix = co_matrix.tocsc()
        pkl.dump(co_matrix, file=open(co_matrix_path, 'wb'), )
    else:
        print('loading co-matrix...')
        co_matrix = pkl.load(file=open(co_matrix_path, 'rb'))

    if not os.path.exists(inst2idx_dict_path):
        inst2idx_dict = dict()
        with open(inst2idx_path, 'r') as r:
            for i, line in enumerate(tqdm(r)):
                instance, idx= line.split('\t')
                inst2idx_dict[instance] = int(idx)
        pkl.dump(inst2idx_dict, file=open(inst2idx_dict_path, 'wb'), )

    else:
        print('loading inst2idx_dict...')
        inst2idx_dict = pkl.load(file=open(inst2idx_dict_path, 'rb'))

    vec_data = []
    for line in data:
        tmp = csc_matrix((len(line), concept_num))
        for i, word in enumerate(line):
            idx = inst2idx_dict.get(word)
            if not idx:
                continue
            tmp[i] = co_matrix[:, idx].T
        vec_data.append(tmp)

    return vec_data


if __name__ == '__main__':
    pass