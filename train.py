import os
from ultralytics.utils import yaml_load
from ultralytics import YOLO
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    cfg = yaml_load("train_val_test_cfg.yaml")

    dataset_path = os.path.join(ROOT, 'datasets', cfg['dataset'], 'dataset.yaml')
    output_dir = os.path.join(ROOT, 'train_out', cfg['project'])

    if resume_chkpt:= cfg['train_cfg']['resume_chkpt']:
        chkpt = os.path.join(output_dir, cfg['name'] , "weights", resume_chkpt)
        model = YOLO(model=chkpt).train(resume = True)
        print(f"[{os.path.basename(__file__)}]\tResume training from {chkpt}")
    else:
        #pretrain weights:
        if pretrain_chkpt:= cfg['train_cfg']['pretrain_chkpt']:
            #pretrain_weight = r"F:\MIS_data\YOLO\train_out\MIS_no_pass\trial_1\weights\best.pt"
            model = YOLO(pretrain_chkpt) # Use a pretrain model
            print(f"[{os.path.basename(__file__)}]\tLoaded pretrained weight at {pretrain_chkpt}")
        #no pretrain weights:
        else:
            model_cfg_path = os.path.join(ROOT, 'cfg', 'det', cfg['train_cfg']['model_cfg_yaml'])
            model = YOLO(model_cfg_path, task= 'detect')  # build a new model from YAML
            print(f"[{os.path.basename(__file__)}]\tStart training from scratch at {output_dir} ")

        results = model.train(data=dataset_path, project = output_dir, name = cfg['name'], **cfg['train_cfg']['train_param'])
        print(f"[{os.path.basename(__file__)}]\tCompleted training at {output_dir}")

    shutil.copy2("train_cfg.yaml", os.path.join(output_dir, cfg['name']))