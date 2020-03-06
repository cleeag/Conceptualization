import os
from os.path import join

def count_match(label_true, label_pred):
    cnt = 0
    for l in label_true:
        if l in label_pred:
            cnt += 1
    return cnt

def microf1(true_labels_dict, pred_labels_dict):
    # assert len(true_labels_dict) == len(pred_labels_dict)
    l_true_cnt, l_pred_cnt, hit_cnt = 0, 0, 0
    for mention_id, labels_true in true_labels_dict.items():
        if mention_id not in pred_labels_dict: continue
        labels_pred = pred_labels_dict[mention_id]
        hit_cnt += count_match(labels_true, labels_pred)
        l_true_cnt += len(labels_true)
        l_pred_cnt += len(labels_pred)
    p = hit_cnt / l_pred_cnt
    r = hit_cnt / l_true_cnt
    return 2 * p * r / (p + r + 1e-7)


def macrof1(true_labels_dict, pred_labels_dict, return_pnr=False):
    # assert len(true_labels_dict) == len(pred_labels_dict)
    p_acc, r_acc = 0, 0
    for mention_id, labels_true in true_labels_dict.items():
        if mention_id not in pred_labels_dict: continue
        labels_pred = pred_labels_dict[mention_id]
        if len(labels_true) == 0 or len(labels_pred) == 0:
            continue
        match_cnt = count_match(labels_true, labels_pred)
        p_acc += match_cnt / len(labels_pred)
        r_acc += match_cnt / len(labels_true)
    p, r = p_acc / len(pred_labels_dict), r_acc / len(true_labels_dict)
    f1 = 2 * p * r / (p + r + 1e-7)
    if return_pnr:
        return f1, p, r
    else:
        return f1



if __name__ == '__main__':
    pass