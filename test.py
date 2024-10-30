import os
from ultralytics import YOLO
import cv2
import csv
from tqdm import tqdm
import shutil
from utils.format import round_speeds, format_labels
from ultralytics.utils import yaml_load
ROOT = os.path.dirname(os.path.abspath(__file__))

RESULT_HEADER = ['img_path', 'preprocess', 'inference', 'post process', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class_id']
CONFIG_YAML = "train_val_test_cfg.yaml"

if __name__ == '__main__':

    cfg = yaml_load(CONFIG_YAML)

    output_dir = os.path.join(ROOT, 'test_out', cfg['project'])

    # Load a model
    model_chktp_path = os.path.join(ROOT, 'train_out', cfg['project'], cfg['name'], "weights", cfg['test_cfg']['model_chkpt'])
    model = YOLO(model_chktp_path)

    #Data path
    img_path = os.path.join(ROOT, 'datasets', cfg['dataset'], 'images', cfg['test_cfg']['test_param']['split'])
    image_files = [file for file in os.listdir(img_path) if file.endswith('.jpg')]
    #Anno path
    anno_path = img_path.replace("images", "labels")
    anno_files = [file for file in os.listdir(anno_path) if file.endswith('.txt')]
    #TODO: load all GT_lbls into a dict


    print(f"[{os.path.basename(__file__)}]\tTesting chkpt path: {model_chktp_path}")
    print(f"[{os.path.basename(__file__)}]\tValidation on img_path: {img_path}")
    metrics = model.val(
        project = output_dir,
        name = cfg['name'],
        **cfg['test_cfg']['test_param']
    )
    #print(f"fitness: {metrics.fitness}")
    #print(metrics.results_dict)
    #print(metrics.speed)
    
    print(f"[{os.path.basename(__file__)}]\tPrediction on img_path: {img_path}")
    

    out_dir = os.path.join(output_dir, cfg['name'])
    out_dir_img = os.path.join(out_dir, 'imgs')
    out_dir_lbls = os.path.join(out_dir, 'labels')
    out_dir_crop = os.path.join(out_dir, 'crops')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)
    os.makedirs(out_dir_lbls, exist_ok=True)
    os.makedirs(out_dir_crop, exist_ok=True)

    test_summary = os.path.join(out_dir, 'prediction_summary.csv')
    with open(test_summary, 'w', encoding= 'utf-8', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_HEADER)

        print(f"[{os.path.basename(__file__)}]\tProceed to perform prediction:")
        for img in tqdm(image_files):
            try:
                prediction_summary = []
                image_path = os.path.join(img_path, img)
                save_path_img = os.path.join(out_dir_img, img)
                save_path_lbl = os.path.join(out_dir_lbls, img.replace(".jpg", '.txt'))
                results = model.predict(
                    source=image_path,
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

    print(f"[{os.path.basename(__file__)}]\tVis result for dataset: '{img_path} saved in {out_dir}'")
    shutil.copy2(CONFIG_YAML, os.path.join(output_dir, cfg['name']))

    #TODO: save TP, TN, FP, FN for analysis

    #TODO: load all pred_lbl
    #GT_lbls = load_lbls()
    for anno_file in anno_files:
        anno_path = os.path.join(anno_path, anno_file)
        #GT_lbls = load_lbl(anno_path)
        #load prediction image
        #save to corresponding
    '''
    out_dir_TP = os.path.join(out_dir, confusion_matrix, 'TP')
    out_dir_FP = os.path.join(out_dir, confusion_matrix, 'FP')
    out_dir_FN = os.path.join(out_dir, confusion_matrix, 'FN')
    out_dir_TN = os.path.join(out_dir, confusion_matrix, 'TN')
    os.makedirs(out_dir_TP, exist_ok=True)
    os.makedirs(out_dir_FP, exist_ok=True)
    os.makedirs(out_dir_FN, exist_ok=True)
    os.makedirs(out_dir_TN, exist_ok=True)
    '''


