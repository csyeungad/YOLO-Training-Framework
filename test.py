import os
from ultralytics import YOLO
import cv2
import csv
from tqdm import tqdm
import shutil
from utils.format import round_speeds, format_labels
from utils.dataset import load_classify_img_paths, load_detect_img_paths
from utils.analysis import get_CM_grouping
from ultralytics.utils import yaml_load
ROOT = os.path.dirname(os.path.abspath(__file__))

DETECT_RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class_id']
CLASSIFY_RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'top1', 'top1conf', 'top3', 'top3conf']
CONFIG_YAML = "train_val_test_cfg.yaml"

if __name__ == '__main__':

    cfg = yaml_load(CONFIG_YAML)
    print(cfg)
    task = cfg['task'].lower()
    output_dir = os.path.join(ROOT, 'test_out', task , cfg['project'])

    # Load a model
    model_chktp_path = os.path.join(ROOT, 'train_out', task ,  cfg['project'], cfg['name'], "weights", cfg['test_cfg']['model_chkpt'])
    model = YOLO(model_chktp_path)
    if not os.path.exists(os.path.join(output_dir, cfg['name'])):
        os.makedirs(os.path.join(output_dir, cfg['name']))
    shutil.copy2(CONFIG_YAML, os.path.join(output_dir, cfg['name']))

    print(f"[{os.path.basename(__file__)}]\tTesting chkpt path: {model_chktp_path}")
    metrics = model.val(
        project = output_dir,
        name = cfg['name'],
        **cfg['test_cfg']['test_param']
    )

    #Data path
    dataset_path = os.path.join(ROOT, 'datasets', cfg['dataset'])
    if task == 'detect':
        data_path = os.path.join(dataset_path, 'images', cfg['test_cfg']['test_param']['split'])
        img_paths = load_detect_img_paths(data_path)
        img_ext = os.path.splitext(img_paths[0])[1]
        #Anno path
        anno_path = data_path.replace("images", "labels")
        anno_files = [file for file in os.listdir(anno_path) if file.endswith('.txt')]
    
    if task == 'classify':
        data_path = os.path.join(dataset_path,  cfg['test_cfg']['test_param']['split'])
        img_paths = load_classify_img_paths(data_path)
        img_ext = os.path.splitext(img_paths[0])[1]

    print(f"[{os.path.basename(__file__)}]\tPrediction on data_path: {data_path}")
    if task == 'detect':
        out_dir = os.path.join(output_dir, cfg['name'])
        out_dir_img = os.path.join(out_dir, 'pred_imgs')
        out_dir_lbls = os.path.join(out_dir, 'pred_labels')
        out_dir_crop = os.path.join(out_dir, 'pred_crops')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)
        os.makedirs(out_dir_lbls, exist_ok=True)
        os.makedirs(out_dir_crop, exist_ok=True)

        test_summary = os.path.join(out_dir, 'prediction_summary.csv')
        with open(test_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
            writer = csv.writer(f)
            writer.writerow(DETECT_RESULT_HEADER)

            print(f"[{os.path.basename(__file__)}]\tProceed to perform prediction:")
            for img_path in tqdm(img_paths):
                try:
                    prediction_summary = []
                    img = os.path.basename(img_path)
                    save_path_img = os.path.join(out_dir_img, img)
                    save_path_lbl = os.path.join(out_dir_lbls, img.replace(".jpg", '.txt'))
                    results = model.predict(
                        source=img_path,
                        **cfg['test_cfg']['test_param']
                    )

                    #print(results[0].boxes)
                    prediction_summary = [results[0].path]
                    prediction_summary.extend(round_speeds(results[0].speed.values()))
                    prediction_summary.extend(format_labels(results[0].boxes.data.tolist()))
                    writer.writerow(prediction_summary)
                    

                    #annotated_image = results[0].plot()
                    results[0].save_txt(save_path_lbl)
                    results[0].save_crop(save_dir=out_dir_crop, file_name=img)
                    results[0].save(save_path_img)
                except Exception as e:
                    print(e)
        
        if cfg['test_cfg']['CM_grouping_vis']:
            print(f"[{os.path.basename(__file__)}]\tProceed to perform confusion matrix grouping...")
            TN, FN, FP, TP = get_CM_grouping(out_dir_lbls, anno_path)

            cm = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
            base_out_dir = os.path.join(out_dir, 'confusion_matrix')

            for label, images in cm.items():
                out_dir_label = os.path.join(base_out_dir, label)
                os.makedirs(out_dir_label, exist_ok=True)

                for img in images:
                    src_path = os.path.join(out_dir_img, f"{img}{img_ext}")
                    dest_path = os.path.join(out_dir_label, f"{img}{img_ext}")
                    shutil.copy2(src_path, dest_path)

    if task == 'classify':
        out_dir = os.path.join(output_dir, cfg['name'])
        out_dir_img = os.path.join(out_dir, 'pred_imgs')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_img, exist_ok=True)

        test_summary = os.path.join(out_dir, 'prediction_summary.csv')
        with open(test_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
            writer = csv.writer(f)
            writer.writerow(CLASSIFY_RESULT_HEADER)

            print(f"[{os.path.basename(__file__)}]\tProceed to perform prediction:")
            for img_path in tqdm(img_paths):
                try:
                    prediction_summary = []
                    img = os.path.basename(img_path)
                    save_path_img = os.path.join(out_dir_img, img)
                    results = model.predict(
                        source=img_path,
                        **cfg['test_cfg']['test_param']
                    )
                    #print(results[0].probs)
                    prediction_summary = [results[0].path]
                    prediction_summary.extend(round_speeds(results[0].speed.values()))
                    prediction_summary.append(results[0].probs.top1)
                    prediction_summary.append(results[0].probs.top1conf.tolist())
                    prediction_summary.extend(results[0].probs.top5[:3])
                    prediction_summary.extend(results[0].probs.top5conf.tolist()[:3])
                    writer.writerow(prediction_summary)
    
                    results[0].save(save_path_img)
                except Exception as e:
                    print(e)

    print(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{data_path} saved in {out_dir}'")
    




