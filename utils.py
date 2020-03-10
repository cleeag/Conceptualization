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

from metrics import macrof1, microf1

black_list = ['it', 'hi', 'a', 'the', 'wa', 'thi', 'i', 'she', 'them', 'we', 'us']


def read_data(path):
    data = []
    with open(path, 'r') as r:
        for i, line in enumerate(r):
            data.append(line.split())
    return data


def read_ufet_data(path):
    json_data = []
    raw_data = []
    with open(path, 'r') as r:
        count = 0
        for i, line in enumerate(r):
            if count >= 100: break
            example = json.loads(line.strip())
            json_data.append(example)
            # line = example['left_context_token'] + example['mention_span'].split() + example['right_context_token']
            line = example['mention_span'].split()
            annot_id = example['annot_id']
            if len(line) > 0:
                raw_data.append((annot_id, line))
                count += 1
    return json_data, raw_data


def export_result_file(raw_data, term_data, annot_ids, idx2concept_dict, result_path, output_path):
    C = pkl.load(file=open(result_path, 'rb'))
    with open(output_path, 'w') as w:
        for idx, c_opt in enumerate(C):
            top_5 = c_opt.toarray().flatten().argsort()[-5:][::-1]
            print(idx)
            print(raw_data[idx])
            print(term_data[idx])
            w.write(f'({str(idx)})\n')
            w.write(f'({annot_ids})\n')
            w.write(f'{raw_data[idx]}\n')
            w.write(f'{term_data[idx]}\n')
            for x in top_5:
                w.write('\t'.join([str(x), str(c_opt[x]), idx2concept_dict[x]]) + '\n')
                print(x, c_opt[x], idx2concept_dict[x])
            print()
            w.write('\n')


def calculate_f1(C_path, idx2concept_dict_path, annot_ids_path, ufet_path):
    C = pkl.load(file=open(C_path, 'rb'))
    idx2concept_dict = pkl.load(file=open(idx2concept_dict_path, 'rb'))
    annot_ids = pkl.load(file=open(annot_ids_path, 'rb'))
    pred_dict = {}
    # C = C[:3]
    # print(C.shape)
    for idx, c_opt in tqdm(enumerate(C)):
        top_5 = c_opt.toarray().flatten().argsort()[-5:][::-1]
        top_5_concept = [idx2concept_dict[x] for x in top_5]
        pred_dict[annot_ids[idx]] = top_5_concept

    ufet_dict = {}
    with open(ufet_path, 'r') as r:
        for line in r:
            example = json.loads(line.strip())
            annot_id = example['annot_id']
            if annot_id not in pred_dict: continue
            ans = example['y_str']
            ufet_dict[annot_id] = ans
    print(len(pred_dict), len(ufet_dict))
    maf1 = macrof1(ufet_dict, pred_dict, return_pnr=False)
    mif1 = microf1(ufet_dict, pred_dict)

    print(maf1, mif1)
    return maf1, mif1


def co_occurence_lookup(data,
                        concept_num,
                        instance_num,
                        annot_ids_path,
                        one_word_per_term=False,
                        clean_word=True,
                        raw_file_dir_path='',
                        co_matrix_path='',
                        inst2idx_dict_path='',
                        idx2concept_dict_path=''):
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
        inst2idx_dict = defaultdict(dict)
        with open(join(raw_file_dir_path, 'instanceFile'), 'r') as r:
            for i, line in enumerate(tqdm(r)):
                instance, idx = line.split('\t')
                instance = tuple(instance.lower().split())
                if instance[0] in inst2idx_dict:
                    if instance in inst2idx_dict[instance[0]]:
                        inst2idx_dict[instance[0]][instance].append(int(idx))
                    else:
                        inst2idx_dict[instance[0]][instance] = [int(idx)]
                else:
                    inst2idx_dict[instance[0]][instance] = [int(idx)]

                if i == 0: print(inst2idx_dict)

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

    vec_data, term_data, sent_data, annot_ids = [], [], [], []
    p = inflect.engine()
    for (annot_id, line) in tqdm(data):
        tmp = []
        term_tmp = []
        if clean_word:
            clean_line = []
            for w in line:
                word = ''.join(e for e in w if e.isalnum())
                singular_word = p.singular_noun(word)
                word = singular_word if singular_word else word
                word = word.lower()
                clean_line.append(word)
            line = clean_line

        if one_word_per_term:
            for i, word in enumerate(line):
                idx = inst2idx_dict.get(word)
                if not idx: continue
                tmp.append([co_matrix[:, idx].T])
        else:
            i = 0
            while i < len(line):
                word = line[i]
                if word in inst2idx_dict:
                    max_len = 1
                    candidate_dict = {}
                    for term in inst2idx_dict[word]:
                        if tuple(line[i:i + len(term)]) == term:
                            if len(term) >= max_len:
                                max_len = len(term)
                                candidate_dict[max_len] = term
                    if len(candidate_dict) == 0 or (
                            candidate_dict[max_len][0] in black_list and len(candidate_dict) == 1):
                        i += 1
                    elif len(candidate_dict) > 0:
                        term = candidate_dict[max_len]
                        idx = inst2idx_dict[word][term]
                        # print(idx, term, line)
                        occ_vec = csc_matrix(co_matrix[:, idx].sum(axis=1).T)
                        # tmp.append([co_matrix[:, idx].T])
                        tmp.append([occ_vec])
                        term_tmp.append(term)
                        i += max_len
                else:
                    i += 1
        if len(tmp) > 0:
            tmp = bmat(tmp, format='csc')
            vec_data.append(tmp)
            sent_data.append(line)
            term_data.append(term_tmp)
            annot_ids.append(annot_id)
            # print(sent_data, term_data)
    pkl.dump(annot_ids, file=open(annot_ids_path, 'wb'))
    return vec_data, sent_data, term_data, annot_ids, idx2concept_dict


if __name__ == '__main__':
    test_data = '/home/data/cleeag/conceptualization/test_data.txt'
    raw_file_dir_path = '/home/data/cleeag/conceptualization/short-text-conceptualization'
    co_matrix_path = '/home/data/cleeag/conceptualization/co_matrix.pkl'
    # inst2idx_dict_path = '/home/data/cleeag/conceptualization/inst2idx_dict.pkl'
    inst2idx_simple_dict_path = '/home/data/cleeag/conceptualization/inst2idx_dict-simple.pkl'
    idx2concept_dict_path = '/home/data/cleeag/conceptualization/idx2concept_dict.pkl'
    C_result_path = '/home/data/cleeag/conceptualization/result/C.pkl'
    annot_ids_path = '/home/data/cleeag/conceptualization/result/annot_id.pkl'
    ufet_path = '/home/data/cleeag/ufet_crowd/dev.json'

    concept_num = 2359855
    instance_num = 6215859

    calculate_f1(C_result_path, idx2concept_dict_path, annot_ids_path, ufet_path)
    sys.exit()
    raw_data = read_data(test_data)
    vec_data, sent_data, term_data, annot_ids, idx2concept_dict = co_occurence_lookup(raw_data, concept_num,
                                                                                      instance_num, annot_ids_path,
                                                                                      raw_file_dir_path=raw_file_dir_path,
                                                                                      co_matrix_path=co_matrix_path,
                                                                                      inst2idx_dict_path=inst2idx_simple_dict_path,
                                                                                      idx2concept_dict_path=idx2concept_dict_path)
