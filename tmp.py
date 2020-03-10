import os
from os.path import join
import itertools
import pickle as pkl
from scipy.sparse import *
from tqdm import tqdm
import sys
from collections import defaultdict
import json

import inflect

def check_dict(inst2idx_dict_path):
    idx2concept_dict = pkl.load(file=open(inst2idx_dict_path, 'rb'))
    while True:
        key = input('please input key\n')
        print(idx2concept_dict.get(key, 'not found'))


def instance2concept_lookup(raw_file_dir_path='',
                            co_matrix_path='',
                            inst2idx_dict_path='',
                            idx2concept_dict_path=''):
    print('loading co-matrix...')
    co_matrix = pkl.load(file=open(co_matrix_path, 'rb'))

    print('loading inst2idx_dict...')
    inst2idx_dict = {}
    if not os.path.exists(inst2idx_dict_path):
    # if True:
        with open(join(raw_file_dir_path, 'instanceFile'), 'r') as r:
            for i, line in enumerate(tqdm(r)):
                # instance, idx = line.lower().split('\t')
                instance, idx = line.split('\t')
                if instance in inst2idx_dict: print(instance)
                inst2idx_dict[instance] = int(idx)

        pkl.dump(inst2idx_dict, file=open(inst2idx_dict_path, 'wb'), )
    else:
        inst2idx_dict = pkl.load(file=open(inst2idx_dict_path, 'rb'))

    print('loading idx2concept_dict...')
    idx2concept_dict = pkl.load(file=open(idx2concept_dict_path, 'rb'))

    while True:
        key = input('\nplease input instance\n')
        key = inst2idx_dict.get(key)
        if not key:
            print('not found')
            continue
        c_vector = co_matrix[:, key].T.toarray()
        top_5 = c_vector.flatten().argsort()[-5:][::-1]
        for i in range(len(top_5)):
            print(idx2concept_dict[top_5[i]], c_vector[0][top_5[i]])


if __name__ == '__main__':
    test_data = '/home/data/cleeag/conceptualization/test_data.txt'
    raw_file_dir_path = '/home/data/cleeag/conceptualization/short-text-conceptualization'
    co_matrix_path = '/home/data/cleeag/conceptualization/co_matrix.pkl'
    # inst2idx_dict_path = '/home/data/cleeag/conceptualization/inst2idx_dict.pkl'
    inst2idx_simple_dict_path = '/home/data/cleeag/conceptualization/inst2idx_dict-simple.pkl'
    idx2concept_dict_path = '/home/data/cleeag/conceptualization/idx2concept_dict.pkl'
    concept_num = 2359855
    instance_num = 6215859

    # check_dict(inst2idx_dict_path)
    instance2concept_lookup(raw_file_dir_path, co_matrix_path, inst2idx_simple_dict_path, idx2concept_dict_path)