from ultralytics import YOLO
from ultralytics.utils import yaml_load
CONFIG_YAML = "export_cfg.yaml"

if __name__ == '__main__':
    cfg = yaml_load(CONFIG_YAML)
    model = YOLO(model=cfg['export_cfg']['model_chkpt'])
    model.export(**cfg['export_cfg']['export_param']) #onnx torchscript engine

    







