import os
from ultralytics import YOLO
import cv2
import csv
from tqdm import tqdm
import shutil
from utils.format import round_speeds, format_labels
from utils.dataset import load_classify_img_paths, load_detect_img_paths
from utils.analysis import get_CM_grouping, get_top_cls_distribution, compute_metrics
from ultralytics.utils import yaml_load
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
DETECT_RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class_id']
CLASSIFY_RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'top1', 'top1conf', 'top3', 'top3conf']
CONFIG_YAML = "infer_cfg.yaml"

if __name__ == '__main__':

    cfg = yaml_load(CONFIG_YAML)
    (f"[{os.path.basename(__file__)}]\tCfg: {cfg}")
    task = cfg['task'].lower()
    output_dir = os.path.join(ROOT, 'infer_out', task , cfg['project'])
    dataset_path = os.path.join(ROOT, 'datasets', cfg['dataset'])

    #Data path
    if task == 'detect':
        data_path = os.path.join(dataset_path, 'images')
        if not os.path.isdir(data_path):
            raise FileNotFoundError('Please ensure a correct dataset structure for inference')
        img_paths = load_detect_img_paths(data_path)
        if not img_paths:
            raise FileNotFoundError('Please ensure a correct dataset structure for inference')
        img_ext = os.path.splitext(img_paths[0])[1]

        #Anno path
        anno_path = data_path.replace("images", "labels")
        anno_files = None
        if os.path.isdir(anno_path) and len(anno_path)>0:
            anno_files = [file for file in os.listdir(anno_path) if file.endswith('.txt')]
        else:
            anno_path = 'No Ground-Truth labels'

    if task == 'classify':
        img_paths = load_classify_img_paths(dataset_path)
        img_ext = os.path.splitext(img_paths[0])[1]

    #chkpt path
    model_chktp_path = cfg['infer_cfg']['model_chkpt']
    model = YOLO(model_chktp_path)
    if not os.path.isdir(os.path.join(output_dir, cfg['name'])):
        os.makedirs(os.path.join(output_dir, cfg['name']))

    logging.basicConfig(
        filename= os.path.join(output_dir, cfg['name'], f"infer_log.log"),
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - %(message)s',
        filemode= 'w'
    )

    print(f"[{os.path.basename(__file__)}]\tInference chkpt path: {model_chktp_path}")
    print(f"[{os.path.basename(__file__)}]\tInference data path: {dataset_path}")
    print(f"[{os.path.basename(__file__)}]\tGround-Truth anno path: {anno_path}")
    logging.info(f"[{os.path.basename(__file__)}]\tInference chkpt path: {model_chktp_path}")
    logging.info(f"[{os.path.basename(__file__)}]\tInference data path: {dataset_path}")
    logging.info(f"[{os.path.basename(__file__)}]\tGround-Truth anno path: {anno_path}")

    if task == 'detect':
        out_dir = os.path.join(output_dir, cfg['name'])
        out_dir_img = os.path.join(out_dir, 'pred_imgs')
        out_dir_lbls = os.path.join(out_dir, 'pred_labels')
        out_dir_crop = os.path.join(out_dir, 'pred_crops')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)
        os.makedirs(out_dir_lbls, exist_ok=True)
        os.makedirs(out_dir_crop, exist_ok=True)

        print(f"Inference parameters:{cfg['infer_cfg']['infer_param']}")
        logging.info(f"Inference parameters:{cfg['infer_cfg']['infer_param']}")

        infer_summary = os.path.join(out_dir, 'prediction_results.csv')
        with open(infer_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
            writer = csv.writer(f)
            writer.writerow(DETECT_RESULT_HEADER)

            print(f"[{os.path.basename(__file__)}]Proceed to perform prediction:")
            
            for img_path in tqdm(img_paths):
                try:
                    prediction_results = []
                    img = os.path.basename(img_path)
                    save_path_img = os.path.join(out_dir_img, img)
                    save_path_lbl = os.path.join(out_dir_lbls, img.replace(".jpg", '.txt'))
                    results = model.predict(
                        source=img_path,
                        **cfg['infer_cfg']['infer_param']
                    )

                    prediction_results = [results[0].path]
                    prediction_results.extend(round_speeds(results[0].speed.values()))
                    prediction_results.extend(format_labels(results[0].boxes.data.tolist()))
                    writer.writerow(prediction_results)

                    annotated_image = results[0].plot()
                    results[0].save_txt(save_path_lbl)
                    if cfg['infer_cfg']['save_crop']:
                        results[0].save_crop(save_dir=out_dir_crop, file_name=img)
                    results[0].save(save_path_img)
                except Exception as e:
                    print(e)

        logging.info(f"defect class id name:\n{model.names}")
        top_cls_dist = get_top_cls_distribution(infer_summary, model.names)
        top_cls_dist_log = (f"\nPrediction top_cls_distribution:\n")
        for cls, count in top_cls_dist.items():
            top_cls_dist_log += f"{cls}:{count}\n"
        print(top_cls_dist_log)
        logging.info(top_cls_dist_log)
        

        if cfg['infer_cfg']['CM_grouping_vis'] and anno_files:
            print(f"[{os.path.basename(__file__)}]\tProceed to perform confusion matrix grouping...")
            logging.info(f"[{os.path.basename(__file__)}]\tProceed to perform confusion matrix grouping...")
            TN, FN, FP, TP = get_CM_grouping(out_dir_lbls, anno_path)
            accuracy, precision, recall, f1_score = compute_metrics(len(TP), len(TN), len(FP), len(FN))
            total_img_num = len(img_paths)

            tn_percent = round(len(TN) / total_img_num * 100, 3)
            fn_percent = round(len(FN) / total_img_num * 100, 3)
            fp_percent = round(len(FP) / total_img_num * 100, 3)
            tp_percent = round(len(TP) / total_img_num * 100, 3)

            CM_log  = (f"\nConfusion Matrix:\nTP: {len(TP)} ({tp_percent}%)\tTN: {len(TN)} ({tn_percent}%)\tFP: {len(FP)} ({fp_percent}%)\tFN: {len(FN)} ({fn_percent}%)")
            metrics_log = (f"\nAccuracy:\t{accuracy:.3f}\tPrecision:\t{precision:.3f}\tRecall:\t{recall:.3f}\tF1 Score:\t{f1_score:.3f}\n")
            
            print(CM_log)
            print(metrics_log)
            logging.info(CM_log)
            logging.info(metrics_log)

            cm = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
            base_out_dir = os.path.join(out_dir, 'confusion_matrix')

            # Create directories and copy files
            for label, images in cm.items():
                    if label in ["FP", "FN"]:
                        out_dir_label = os.path.join(base_out_dir, label)
                        os.makedirs(out_dir_label, exist_ok=True)
                        for img in images:
                            try:
                                src_path = os.path.join(out_dir_img, f"{img}{img_ext}")
                                dest_path = os.path.join(out_dir_label, f"{img}{img_ext}")
                                shutil.copy2(src_path, dest_path)
                            except Exception as e:
                                print(e)

    if task == 'classify':
        out_dir = os.path.join(output_dir, cfg['name'])
        out_dir_img = os.path.join(out_dir, 'pred_imgs')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)

        test_summary = os.path.join(out_dir, 'prediction_results.csv')
        with open(test_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
            writer = csv.writer(f)
            writer.writerow(CLASSIFY_RESULT_HEADER)

            print(f"[{os.path.basename(__file__)}]\tProceed to perform prediction:")
            for img_path in tqdm(img_paths):
                try:
                    prediction_results = []
                    img = os.path.basename(img_path)
                    save_path_img = os.path.join(out_dir_img, img)
                    results = model.predict(
                        source=img_path,
                        **cfg['infer_cfg']['infer_param']
                    )
                    #print(results[0].probs)
                    prediction_results = [results[0].path]
                    prediction_results.extend(round_speeds(results[0].speed.values()))
                    prediction_results.append(results[0].probs.top1)
                    prediction_results.append(results[0].probs.top1conf.tolist())
                    prediction_results.extend(results[0].probs.top5[:3])
                    prediction_results.extend(results[0].probs.top5conf.tolist()[:3])
                    writer.writerow(prediction_results)
    
                    results[0].save(save_path_img)
                except Exception as e:
                    print(e)

    print(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{dataset_path} saved in {out_dir}'")
    print(f"[{os.path.basename(__file__)}]\tCfg file archived to: '{os.path.join(output_dir, cfg['name'])}'")
    logging.info(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{dataset_path} saved in {out_dir}'")
    logging.info(f"[{os.path.basename(__file__)}]\tCfg file archived to: '{os.path.join(output_dir, cfg['name'])}'")
    shutil.copy2(CONFIG_YAML, os.path.join(output_dir, cfg['name']))

