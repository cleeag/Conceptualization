import os
from os.path import join
import itertools
import pickle as pkl
from scipy.sparse import *
from tqdm import tqdm
import sys

import inflect

def read_data(path):
    data = []
    with open(path, 'r') as r:
        for i, line in enumerate(r):
            data.append(line.split())
    return data


def co_occurence_lookup(data, concept_num, instance_num, raw_file_dir_path='',
                        co_matrix_path='', inst2idx_dict_path='', idx2concept_dict_path=''):
    if not os.path.exists(co_matrix_path):
        co_matrix = lil_matrix((concept_num, instance_num))
        with open(join(raw_file_dir_path, 'conceptInstanceFile'), 'r') as r:
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
        with open(join(raw_file_dir_path, 'instanceFile'), 'r') as r:
            for i, line in enumerate(tqdm(r)):
                instance, idx = line.split('\t')
                inst2idx_dict[instance] = int(idx)
        pkl.dump(inst2idx_dict, file=open(inst2idx_dict_path, 'wb'), )
    else:
        print('loading inst2idx_dict...')
        inst2idx_dict = pkl.load(file=open(inst2idx_dict_path, 'rb'))

    if not os.path.exists(idx2concept_dict_path):
        idx2concept_dict = dict()
        with open(join(raw_file_dir_path, 'conceptFile'), 'r') as r:
            for i, line in enumerate(tqdm(r)):
                concept, idx = line.split('\t')
                idx2concept_dict[int(idx)] = concept
        pkl.dump(idx2concept_dict, file=open(idx2concept_dict_path, 'wb'), )
    else:
        print('loading idx2concept_dict...')
        idx2concept_dict = pkl.load(file=open(idx2concept_dict_path, 'rb'))

    vec_data = []
    p = inflect.engine()
    for line in data:
        tmp=[]
        for i, word in enumerate(line):
            word = ''.join(e for e in word if e.isalnum())
            singular_word = p.singular_noun(word)
            word = singular_word if singular_word else word
            idx = inst2idx_dict.get(word)
            if not idx:
                continue
            tmp.append([co_matrix[:, idx].T])
        tmp = bmat(tmp, format='csc')
        vec_data.append(tmp)

    return vec_data, idx2concept_dict

def check_dict(inst2idx_dict_path):
    idx2concept_dict = pkl.load(file=open(inst2idx_dict_path, 'rb'))
    while True:
        key = input('please input key\n')
        print(idx2concept_dict.get(key, 'not found'))


def inspect_result(raw_data, idx2concept_dict, result_path, output_path):
    C = pkl.load(file=open(result_path, 'rb'))
    with open(output_path, 'w') as w:
        for idx, c_opt in enumerate(C):
            top_3 = c_opt.flatten().argsort()[-5:][::-1]
            print(idx)
            print(raw_data[idx])
            w.write(f'({str(idx)})   {raw_data[idx]}\n')
            for x in top_3:
                w.write('\t'.join([str(x), str(c_opt[x]), idx2concept_dict[x]]) + '\n')
                print(x, c_opt[x], idx2concept_dict[x])
            print()
            w.write('\n')


if __name__ == '__main__':
    test_data = '/home/data/cleeag/conceptualization/test_data.txt'
    raw_file_dir_path = '/home/data/cleeag/conceptualization/short-text-conceptualization'
    co_matrix_path = '/home/data/cleeag/conceptualization/co_matrix.pkl'
    inst2idx_dict_path = '/home/data/cleeag/conceptualization/inst2idx_dict.pkl'
    idx2concept_dict_path = '/home/data/cleeag/conceptualization/idx2concept_dict.pkl'
    concept_num = 2359855
    instance_num = 6215859

    # check_dict(inst2idx_dict_path)
    # sys.exit()

    raw_data = read_data(test_data)
    input_data, idx2concept_dict = co_occurence_lookup(raw_data, concept_num, instance_num,
                                                       raw_file_dir_path=raw_file_dir_path,
                                                       co_matrix_path=co_matrix_path,
                                                       inst2idx_dict_path=inst2idx_dict_path,
                                                       idx2concept_dict_path=idx2concept_dict_path)