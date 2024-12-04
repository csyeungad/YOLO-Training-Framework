import os
from .dataset import load_lbls
import csv
from collections import Counter

def get_CM_grouping(pred_lbl_dir, gt_lbl_dir):
    """ Group the prediction result in to confusion matrix
    
    Return:
        list of img names without ext for each confusion matrix entries
    """

    pred_lbls_dict:dict = load_lbls(pred_lbl_dir)
    gt_lbls_dict:dict = load_lbls(gt_lbl_dir)

    TP = [ gt_file for gt_file in gt_lbls_dict.keys() if len(gt_lbls_dict[gt_file]) == 0 and gt_file not in pred_lbls_dict.keys()] #real pass data not in pred
    TN = [ pred_file for pred_file in pred_lbls_dict.keys() if pred_file in gt_lbls_dict.keys() and len(gt_lbls_dict[pred_file]) > 0] # pred in real defect data
    FP = [ gt_file for gt_file in gt_lbls_dict.keys() if len(gt_lbls_dict[gt_file]) > 0 and gt_file not in pred_lbls_dict.keys()] #real defect data not in pred
    FN = [ pred_file for pred_file in pred_lbls_dict.keys() if pred_file in gt_lbls_dict.keys() and len(gt_lbls_dict[pred_file]) == 0] # pred in real pass data

    return TN, FN, FP, TP

def compute_metrics(TP, TN, FP, FN):
    """Compute accuracy, precision, recall, and F1-score."""
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def get_top_cls_distribution(prediction_file, class_id_name: dict):
    """Compute top class distribution from a prediction summary CSV.
    
    Args:
    prediction_file: the file path of "prediction_results.csv"
    class_id_name: {0: cls_1, 1: cls_2, 2: cls_3,...}
    """
    if os.path.exists(prediction_file):
        names = []
        top_cls = []
        try:
            with open(prediction_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    if len(row) < 10:  # Ensure there are enough columns
                        continue
                    names.append(os.path.basename(row[0]))
                    top_cls.append(row[9])
        
            top_cls_count = Counter(top_cls)
            top_cls_dist = dict(sorted(top_cls_count.items(), key=lambda item: item[1], reverse=True))
            top_cls_dist_name = { class_id_name[int(k)]:v for k,v in top_cls_dist.items()}

            return top_cls_dist_name
        except Exception as e:
            return {}
    else:
        return {}


if __name__ == "__main__":

    top_cls_dist = get_top_cls_distribution(
        prediction_file=r"X:\infer_out\detect\project\name_1\prediction_results.csv",
        class_id_name= {0: 'cls_1', 1: 'cls_2'})
    print(top_cls_dist)