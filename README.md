# YOLO Training Framework

✨ Unified YOLO Framework
A unified framework for seamless training, testing, inference, and deployment of YOLO models, supporting both object detection and classification tasks.

⚙️ Flexible Configuration
Leverages YAML-based configuration files to customize every stage of your YOLO workflow—from dataset preparation to model deployment.

⚡ Streamlined Workflow
Simplify your development process and accelerate results with this end-to-end YOLO solution.

## Dataset Preparation

Datasets must be prepared in the **Ultralytics YOLO format** with proper folder structures to ensure seamless training and inference.

### Object Detection

Refer to the official YOLO documentation on object detection datasets:  
[Ultralytics YOLO Object Detection Dataset Format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)

#### Folder Structure Example:

```
datasets/
└── dataset_name/
    ├── images/
    │   ├── train/
    │   │   ├── img_1.jpg
    │   │   ├── img_2.jpg
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   │   ├── img_1.txt
    │   │   ├── img_2.txt
    │   ├── val/
    │   └── test/
    └── dataset.yaml
```

#### Inference Data Structure:

Labels for inference are **optional**. Example:

```
datasets/
└── dataset_name/
    ├── images/
    │   ├── img_1.jpg
    │   └── img_2.jpg
    └── labels/ (optional)
        ├── img_1.txt
        └── img_2.txt
```

### Classification

Refer to the official YOLO documentation on classification datasets:  
[Ultralytics YOLO Classification Dataset Structure](https://docs.ultralytics.com/datasets/classify/#dataset-structure-for-yolo-classification-tasks)

#### Folder Structure Example:

```
datasets/
└── dataset_name/
    ├── train/
    │   ├── class_1/
    │   │   ├── img_1.jpg
    │   │   └── img_2.jpg
    │   ├── class_2/
    ├── val/
    │   ├── class_1/
    │   └── class_2/
    └── test/
        ├── class_1/
        └── class_2/
```

## Configuration

### Train/Validation/Test Configuration

All training, validation, and testing operations are controlled via the **`train_val_test_cfg.yaml`** file. Customize this file to control dataset paths, model parameters, training hyperparameters, and evaluation settings.

#### Example snippet from `train_val_test_cfg.yaml`:

```yaml
train_cfg:
  resume_chkpt: null #relative path to chkpts to resume or null not too resume 
  pretrain_chkpt: null #relative path to pretrain chkpt or null to train from scratch
  model_cfg_yaml: yolo11s-cls.yaml #model configuration to train from scratch
  train_param: #Refer to default.yaml for all parameters
    #Trainer
    epochs: 3                   # Number of epochs to train for.
    batch: 0.8                  # Specific batch size: batch=16 / Auto mode for % GPU memory utilization (batch= 0.70)
    patience : 20               # Number of epochs to wait without improvement in validation metrics before early stopping the training
    workers : 1
    device: 0                   # device used for training, device = 0,1
    exist_ok : False            # If True, allows overwriting of an existing project/name directory 
    save_period : 20            # Interval for chkpts save
    plots : True                # Save plots and images during train/val
    #Dataset
    imgsz: 320                  # Size of input images as integer, imgze : a -> resize longest size to a, while keeping aspect ratio, imgze : (w, h) -> resize img into square
    rect : False                # True: Pad img after resize to square, False: keep original aspect ratio
    fraction : 0.1              # Train with subset of dataset, 1.0 for full dataset
    single_cls : False          # (bool) train multi-class data as single-class    
    #Optimizer
    optimizer: auto             # Optimizer to use, choices:[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0: 0.01                   # Initial learning rate
    lrf: 0.01                   # Final learning rate as a fraction of the initial rate = (lr0 * lrf)
    momentum: 0.937             # Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update
    weight_decay: 0.0005        # L2 regularization term, penalizing large weights to prevent overfitting.
    #Loss function
    box : 0.1                   # default: 7.5, box loss
    cls : 0.5                   # default: 0.5, classification loss
    dfl : 3.0                   # default: 1.5, distributed focal loss
    multi_scale : False         # (bool) Whether to use multiscale during training
    #Augmentation
    translate : 0.1             # Default: 0.1 , valid: 0.0 - 1.0
    scale : 0.2                 # Default: 0.5 , valid: >=0.0
    flipud: 0.0                 # Default: 0.0 , valid: 0.0 - 1.0
    fliplr : 0.1                # Default: 0.5 , valid: 0.0 - 1.0
    mosaic : 0                  # Combines four training images into one, simulating different scene compositions and object interactions
    close_mosaic : 0

#For validation & prediction in val or testing set
test_cfg: #Refer to default.yaml for all parameters
  model_chkpt: best.pt          #chkpt for testing e.g. best.pt / last.pt        
  CM_grouping_vis: True         # whether to group the prediction result in CM for visualization (Recommended for debugging)
  test_param:
    split: val                  #val / test dataset
    batch: 1                    # Specific batch size: batch=16
    imgsz: 320
    conf: 0.3
    iou: 0.5
    save_json : True
    plots : True
    verbose : True
    exist_ok : True
```

### Inference Configuration

Inference settings can be customized in **`infer_cfg.yaml`**, enabling predictions with or without ground truth labels.

#### Example snippet from `infer_cfg.yaml`:

```yaml
task: classify  #YOLO task, i.e. detect, segment, classify
dataset : imagenet10 
project : imagenet # name the output project
name : imagenet # name the output name
infer_cfg:
  model_chkpt: .\train_out\classify\imagenet\imagenet\weights\best.pt      #relative path for chkpt/deployed model, e.g. .pt, .onnx, .engine .torchscript
  CM_grouping_vis: True         # whether to group the prediction result in CM for visualization if labels are given (Recommended for debugging)
  infer_param:
    imgsz: 640
    conf: 0.1
    iou: 0.1
    save_json : True
    plots : True
    save : False
    verbose : False
    exist_ok : True
    show : False
  save_crop: False
```

### Model Deployment Configuration

To export and deploy a trained checkpoint to various formats, configure the **`export_cfg.yaml`** file.

#### Example snippet from `export_cfg.yaml`:

```yaml
export_cfg:
  model_chkpt: .\train_out\classify\imagenet\imagenet\weights\best.pt      #relative path for chkpt/deployed model, e.g. .pt, .onnx, .engine .torchscript
  export_param:
    format: onnx #e.g. onnx, torchscript, engine
    imgsz: 320 # Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions
    half: false # Enables FP16 quantization, reducing model size and potentially speeding up inference
    int8:  false # 	Enables INT8 quantization, highly beneficial for edge deployments
    dynamic: false # Allows dynamic input sizes for ONNX, TensorRT and OpenVINO exports, enhancing flexibility in handling varying image dimensions.
    device: 0 #Specifies the device for exporting: GPU (device=0), CPU (device=cpu)
```

## Testing and Inference Result Analysis

After testing or inference, the framework generates several folders to help with result analysis and visualization:

- **pred_imgs/**: Visualizations of predictions on images  
- **pred_labels/**: Prediction labels (excluding background)  
- **pred_crop/**: Cropped bounding boxes from predictions  
- **confusion_matrix/**: Confusion matrix visualizations grouping prediction results  

---

## Additional Resources

- [Ultralytics YOLO Docs - Object Detection Dataset Format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)  
- [Ultralytics YOLO Docs - Classification Dataset Structure](https://docs.ultralytics.com/datasets/classify/#dataset-structure-for-yolo-classification-tasks)  


