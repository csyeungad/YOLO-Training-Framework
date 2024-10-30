import os
from ultralytics import YOLO
import cv2
import csv
from tqdm import tqdm
import shutil
from utils.format import round_speeds, format_labels
from utils.dataset import load_lbls
from ultralytics.utils import yaml_load

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class_id']
CONFIG_YAML = "infer_cfg.yaml"

if __name__ == '__main__':

    cfg = yaml_load(CONFIG_YAML)

    #Data path
    dataset = cfg['dataset']
    img_path = os.path.join(ROOT, 'datasets', dataset, 'images')
    if not os.path.isdir(img_path):
        raise FileNotFoundError('Please input a valid dataset')
    img_files = [file for file in os.listdir(img_path) if file.endswith('.jpg')]

    #Anno path
    anno_path = img_path.replace("images", "labels")
    if os.path.isdir(anno_path):
        anno_files = [file for file in os.listdir(anno_path) if file.endswith('.jpg')]
    else:
        anno_path = 'No Ground-Truth labels'

    #chkpt path
    model_chktp_path = cfg['infer_cfg']['model_chkpt']
    
    model = YOLO(model_chktp_path)
    print(f"[{os.path.basename(__file__)}]\tInference chkpt path: {model_chktp_path}")
    print(f"[{os.path.basename(__file__)}]\tInference data path: {img_path}")
    print(f"[{os.path.basename(__file__)}]\tGround-Truth anno path: {anno_path}")

    out_dir = os.path.join(ROOT, 'infer_out', cfg['project'], cfg['name'])
    out_dir_img = os.path.join(out_dir, 'imgs')
    out_dir_lbls = os.path.join(out_dir, 'labels')
    out_dir_crop = os.path.join(out_dir, 'crops')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_lbls, exist_ok=True)
    os.makedirs(out_dir_crop, exist_ok=True)

    infer_summary = os.path.join(out_dir, 'prediction_summary.csv')
    with open(infer_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_HEADER)

        print(f"[{os.path.basename(__file__)}]Proceed to perform prediction:")
        
        for img in tqdm(img_files):
            try:
                prediction_summary = []
                image_path = os.path.join(img_path, img)
                save_path_img = os.path.join(out_dir_img, img)
                save_path_lbl = os.path.join(out_dir_lbls, img.replace(".jpg", '.txt'))
                results = model.predict(
                    source=image_path,
                    **cfg['infer_cfg']['infer_param']
                )

                prediction_summary = [results[0].path]
                prediction_summary.extend(round_speeds(results[0].speed.values()))
                prediction_summary.extend(format_labels(results[0].boxes.data.tolist()))
                writer.writerow(prediction_summary)

                annotated_image = results[0].plot()
                results[0].save_txt(save_path_lbl)
                results[0].save_crop(save_dir=out_dir_crop, file_name=img)
                results[0].save(save_path_img)
            except Exception as e:
                print(e)

    #TODO: Analysis comparision with GT if exist
    pred_lbls_dict:dict = load_lbls(out_dir_lbls)
    gt_lbls_dict:dict = load_lbls(anno_path)
    print(len(pred_lbls_dict), len(gt_lbls_dict))
    FN = [ pred_file for pred_file in pred_lbls_dict.keys() if pred_file in gt_lbls_dict.keys() and len(gt_lbls_dict[pred_file]) > 0] # defect data
    TN = [ pred_file for pred_file in pred_lbls_dict.keys() if pred_file in gt_lbls_dict.keys() and len(gt_lbls_dict[pred_file]) == 0] # pass data
    FP = [ gt_file for gt_file in gt_lbls_dict.keys() if len(gt_lbls_dict[gt_file]) > 0 and gt_file not in pred_lbls_dict.keys()] #defect data not in pred
    TP = [ gt_file for gt_file in gt_lbls_dict.keys() if len(gt_lbls_dict[gt_file]) == 0 and gt_file not in pred_lbls_dict.keys()] #defect data not in pred
    print(len(FN), len(TN), len(FP), len(TP))
    # Compute 
    # FP: pred non exist, gt_exist
    # FN: pred exist, gt exist
    # TN: pred exist, gt exist

    #for anno_path, gt_lbls_ in gt_lbls.items():
    #    for pred_path, pred_lbls_ in pred_lbls.items():

    print(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{img_path} saved in {out_dir}'")
    shutil.copy2(CONFIG_YAML, out_dir)

