import os
from ultralytics import YOLO
import cv2
import csv
from tqdm import tqdm
import shutil
from utils.format import round_speeds, format_labels, format_confusion_matrix
from utils.dataset import load_classify_img_paths, load_detect_img_paths
from utils.analysis import get_CM_grouping, get_top_cls_distribution
from ultralytics.utils import yaml_load
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
DETECT_RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class_id']
CLASSIFY_RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'top1', 'top1conf', 'top3', 'top3conf']
CONFIG_YAML = "train_val_test_cfg.yaml"

if __name__ == '__main__':

    cfg = yaml_load(CONFIG_YAML)
    print(f"[{os.path.basename(__file__)}]\tCfg: {cfg}")
    dataset_path = os.path.join(ROOT, 'datasets', cfg['dataset'])
    dataset_split = cfg['test_cfg']['test_param']['split']
    task = cfg['task'].lower()
    output_dir = os.path.join(ROOT, 'test_out', task , cfg['project'])

    # Load a model
    model_chktp_path = os.path.join(ROOT, 'train_out', task ,  cfg['project'], cfg['name'], "weights", cfg['test_cfg']['model_chkpt'])
    model = YOLO(model_chktp_path)

    print(f"[{os.path.basename(__file__)}]\tTesting chkpt path: {model_chktp_path}")
    
    metrics = model.val(
        project = output_dir,
        name = cfg['name'],
        **cfg['test_cfg']['test_param']
    )
    validation_CM_summary = format_confusion_matrix(metrics)
    logging.basicConfig(
        filename= os.path.join(output_dir, cfg['name'], f"test_log.log"),
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - %(message)s',
        filemode= 'w'
    )
    logging.info(f"[{os.path.basename(__file__)}]\tCfg: {cfg}")
    logging.info(f"[{os.path.basename(__file__)}]\tTesting chkpt path: {model_chktp_path}")
    logging.info(validation_CM_summary)
    logging.info(f"speed: {metrics.speed}")

    #Data path
    if task == 'detect':
        data_path = os.path.join(dataset_path, 'images',dataset_split)
        img_paths = load_detect_img_paths(data_path)
        img_ext = os.path.splitext(img_paths[0])[1] # e.g. .jpg / .bmp 
        #Anno path
        anno_path = data_path.replace("images", "labels")
        anno_files = [file for file in os.listdir(anno_path) if file.endswith('.txt')]
    
    if task == 'classify':
        data_path = os.path.join(dataset_path, dataset_split)
        img_paths = load_classify_img_paths(data_path)
        img_ext = os.path.splitext(img_paths[0])[1]

    print(f"[{os.path.basename(__file__)}]\tPrediction on data_path: {data_path}")
    logging.info(f"[{os.path.basename(__file__)}]\tPrediction on data_path: {data_path}")
    if task == 'detect':
        out_dir = os.path.join(output_dir, cfg['name'])
        out_dir_img = os.path.join(out_dir, 'pred_imgs')
        out_dir_lbls = os.path.join(out_dir, 'pred_labels')
        out_dir_crop = os.path.join(out_dir, 'pred_crops')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)
        os.makedirs(out_dir_lbls, exist_ok=True)
        os.makedirs(out_dir_crop, exist_ok=True)

        print(f"Testing parameters:{cfg['test_cfg']['test_param']}")
        logging.info(f"Testing parameters:{cfg['test_cfg']['test_param']}")

        test_summary = os.path.join(out_dir, 'prediction_results.csv')
        with open(test_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
            writer = csv.writer(f)
            writer.writerow(DETECT_RESULT_HEADER)

            print(f"[{os.path.basename(__file__)}]\tProceed to perform prediction:")
            for img_path in tqdm(img_paths):
                try:
                    prediction_results = []
                    img = os.path.basename(img_path)
                    save_path_img = os.path.join(out_dir_img, img)
                    save_path_lbl = os.path.join(out_dir_lbls, img.replace(".jpg", '.txt'))
                    results = model.predict(
                        source=img_path,
                        **cfg['test_cfg']['test_param']
                    )

                    #print(results[0].boxes)
                    prediction_results = [results[0].path]
                    prediction_results.extend(round_speeds(results[0].speed.values()))
                    prediction_results.extend(format_labels(results[0].boxes.data.tolist()))
                    writer.writerow(prediction_results)
                    

                    #annotated_image = results[0].plot()
                    results[0].save_txt(save_path_lbl)
                    if cfg['infer_cfg']['save_crop']:
                        results[0].save_crop(save_dir=out_dir_crop, file_name=img)
                    results[0].save(save_path_img)
                except Exception as e:
                    print(e)
        
        logging.info(f"defect class id name:\n{model.names}")
        top_cls_dist = get_top_cls_distribution(test_summary, model.names)
        top_cls_dist_log = (f"\nPrediction top_cls_distribution:\n")
        for cls, count in top_cls_dist.items():
            top_cls_dist_log += f"{cls}:{count}\n"
        print(top_cls_dist_log)
        logging.info(top_cls_dist_log)

        if cfg['test_cfg']['CM_grouping_vis']:
            print(f"[{os.path.basename(__file__)}]\tProceed to perform confusion matrix grouping...")
            logging.info(f"[{os.path.basename(__file__)}]\tProceed to perform confusion matrix grouping...")

            TN, FN, FP, TP = get_CM_grouping(out_dir_lbls, anno_path)
            total_img_num = len(img_paths)

            tn_percent = round(len(TN) / total_img_num * 100, 3)
            fn_percent = round(len(FN) / total_img_num * 100, 3)
            fp_percent = round(len(FP) / total_img_num * 100, 3)
            tp_percent = round(len(TP) / total_img_num * 100, 3)

            print(f"\nConfusion Matrix:\nTP: {len(TP)} ({tp_percent}%)\tTN: {len(TN)} ({tn_percent}%)\tFP: {len(FP)} ({fp_percent}%)\tFN: {len(FN)} ({fn_percent}%)\n")
            logging.info(f"\nConfusion Matrix:\nTP: {len(TP)} ({tp_percent}%)\tTN: {len(TN)} ({tn_percent}%)\tFP: {len(FP)} ({fp_percent}%)\tFN: {len(FN)} ({fn_percent}%)\n")

            cm = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
            base_out_dir = os.path.join(out_dir, 'confusion_matrix')

            for label, images in cm.items():
                out_dir_label = os.path.join(base_out_dir, label)
                os.makedirs(out_dir_label, exist_ok=True)

                for img in images:
                    try:
                        #Grouping prediction results to CM folders
                        pred_src_path = os.path.join(out_dir_img, f"{img}{img_ext}")
                        pred_dest_path = os.path.join(out_dir_label, f"{img}{img_ext}")
                        shutil.copy2(pred_src_path, pred_dest_path)
                        #Grouping GT results to CM folders
                        gt_src_path = os.path.join(dataset_path, 'vis', dataset_split ,f"{img}{img_ext}")
                        gt_dest_path = os.path.join(out_dir_label, f"{img}_GT{img_ext}")
                        shutil.copy2(gt_src_path, gt_dest_path)
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
            logging.info(f"[{os.path.basename(__file__)}]\tProceed to perform prediction:")
            for img_path in tqdm(img_paths):
                try:
                    prediction_results = []
                    img = os.path.basename(img_path)
                    save_path_img = os.path.join(out_dir_img, img)
                    results = model.predict(
                        source=img_path,
                        **cfg['test_cfg']['test_param']
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

    print(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{data_path} saved in {out_dir}'")
    logging.info(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{data_path} saved in {out_dir}'")
    print(f"[{os.path.basename(__file__)}]\tCfg file archived to: '{os.path.join(output_dir, cfg['name'])}'")
    logging.info(f"[{os.path.basename(__file__)}]\tCfg file archived to: '{os.path.join(output_dir, cfg['name'])}'")
    shutil.copy2(CONFIG_YAML, os.path.join(output_dir, cfg['name']))




