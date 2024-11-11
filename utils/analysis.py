import os
from .dataset import load_lbls

def get_CM_grouping(pred_lbl_dir, gt_lbl_dir):
    """ Group the prediction result in to confusion matrix"""

    pred_lbls_dict:dict = load_lbls(pred_lbl_dir)
    gt_lbls_dict:dict = load_lbls(gt_lbl_dir)

    # print(f"Pred total: {len(pred_lbls_dict)}")
    # for i, (k, v) in enumerate(pred_lbls_dict.items()):
    #     print(f"\tkey:{k} value:{v}")
    #     if i > 3 : break
    # print(f"GT total: {len(gt_lbls_dict)}")
    # for i, (k, v) in enumerate(gt_lbls_dict.items()):
    #     print(f"\tkey:{k} value:{v}")
    #     if i > 3 : break
    TP = [ gt_file for gt_file in gt_lbls_dict.keys() if len(gt_lbls_dict[gt_file]) == 0 and gt_file not in pred_lbls_dict.keys()] #real pass data not in pred
    TN = [ pred_file for pred_file in pred_lbls_dict.keys() if pred_file in gt_lbls_dict.keys() and len(gt_lbls_dict[pred_file]) > 0] # pred in real defect data
    FP = [ gt_file for gt_file in gt_lbls_dict.keys() if len(gt_lbls_dict[gt_file]) > 0 and gt_file not in pred_lbls_dict.keys()] #real defect data not in pred
    FN = [ pred_file for pred_file in pred_lbls_dict.keys() if pred_file in gt_lbls_dict.keys() and len(gt_lbls_dict[pred_file]) == 0] # pred in real pass data

    return TN, FN, FP, TP

