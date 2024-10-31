import os
from ultralytics.utils import yaml_load
from ultralytics import YOLO
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    cfg = yaml_load("train_val_test_cfg.yaml")

    task = cfg['task'].lower()

    if task == 'detect':
        dataset_path = os.path.join(ROOT, 'datasets', cfg['dataset'], 'dataset.yaml')
    if task == 'classify':
        dataset_path = os.path.join(ROOT, 'datasets', cfg['dataset'])
    print(f"[Dataset path]: {dataset_path}")
    
    output_dir = os.path.join(ROOT,'train_out', task ,cfg['project'])

    if resume_chkpt:= cfg['train_cfg']['resume_chkpt']:
        chkpt = os.path.join(output_dir, cfg['name'] , "weights", resume_chkpt)
        model = YOLO(model=chkpt).train(resume = True)
        print(f"[{os.path.basename(__file__)}]\tResume training from {chkpt}")
    else:
        #pretrain weights:
        if pretrain_chkpt:= cfg['train_cfg']['pretrain_chkpt']:
            model = YOLO(pretrain_chkpt) 
            print(f"[{os.path.basename(__file__)}]\tLoaded pretrained weight at {pretrain_chkpt}")
        #no pretrain weights:
        else:
            model_cfg_path = os.path.join(ROOT, 'cfg', task, cfg['train_cfg']['model_cfg_yaml'])
            model = YOLO(model_cfg_path, task= task)  # build a new model from YAML
            print(f"[{os.path.basename(__file__)}]\tStart training from scratch at {output_dir} ")

        results = model.train(data = dataset_path, project = output_dir, name = cfg['name'], **cfg['train_cfg']['train_param'])
        print(f"[{os.path.basename(__file__)}]\tCompleted training at {output_dir}")

    shutil.copy2("train_val_test_cfg.yaml", os.path.join(output_dir, cfg['name']))
