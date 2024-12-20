## YOLO training framework

All training/testing/prediction/deployment of yolo model in one place.

## Dataset preparation

Prepare the datasets into YOLO format and saved in below folder structures.


### Object detection
Refer to YOLO documentation:
https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

Below shows the snapshot of the folder structure:

    datasets
    |---dataset name
        |---images
        |   |---train
        |   |   |---img_1.jpg
        |   |   |---img_2.jpg
        |   |---val
        |   |---test
        |
        |---labels
        |   |---train
        |   |   |---img_1.txt
        |   |   |---img_2.txt
        |   |---val
        |   |---test
        |
        |---dataset.yaml

Below shows the snapshot for inference data folder structure:

Note: Labels for the inference data is optional

    datasets
    |---dataset name
        |---images
        |   |---img_1.jpg
        |   |---img_2.jpg
        |
        |---labels (optional)
        |   |---img_1.txt
        |   |---img_2.txt



### Classification
Refer to YOLO documentation:
https://docs.ultralytics.com/datasets/classify/#dataset-structure-for-yolo-classification-tasks

Below shows the snapshot of the folder structure:

    datasets
    |---dataset name
        |---train
        |   |---class_1
        |   |   |---img_1.jpg
        |   |   |---img_2.jpg
        |   |---class_2
        |
        |---val
        |   |---class_1
        |   |---class_2
        |
        |---test
        |   |---class_1
        |   |---class_2


## Configuration

### Train \ val \ test Configuration 

All trainings \ testing \ prediction are managed in the **train_val_test_cfg.yaml** config files. Adjust the setting for customised model training \ evaluation \ testing. 

**train_val_test_cfg.yaml**

    task: classify  #YOLO task, i.e. detect, segment, classify
    dataset : imagenet10 
    project : imagenet 
    name : imagenet

    train_cfg:
    resume_chkpt: null # chkpts names to resume or null not too resume  
    pretrain_chkpt: D:/YOLO/train_out/classify/imagenet10/imagenet/weights/best.pt #abs path to pretrain chkpt or null to train from scratch
    model_cfg_yaml: yolo11n-cls.yaml #model configuration to train from scratch
    train_param:
    #Trainer
    epochs: 3                 # Number of epochs to train for.
    batch: 0.8                   # Specific batch size: batch=16 / Auto mode for % GPU memory utilization (batch= 0.70)
    patience : 20               # Number of epochs to wait without improvement in validation metrics before early stopping the training
    workers : 1
    device: 0,1                 # device used for training, device = 0,1
    exist_ok : True             # If True, allows overwriting of an existing project/name directory 
    save_period : 20            # Interval for chkpts save
    plots : True                # Save plots and images during train/val
    #Dataset
    imgsz: 640                  # Size of input images as integer, imgze : a -> resize longest size to a, while keeping aspect ratio, imgze : (w, h) -> resize img into square
    rect : False                # True: Pad img after resize to square, False: keep original aspect ratio
    fraction : 0.3              # Train with subset of dataset, 1.0 for full dataset
    single_cls : False          # (bool) train multi-class data as single-class    
    #Optimizer
    optimizer: auto             # Optimizer to use, choices:[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0: 0.01                   # Initial learning rate
    lrf: 0.01                   # Final learning rate as a fraction of the initial rate = (lr0 * lrf)
    momentum: 0.937             # Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update
    weight_decay: 0.0005        # L2 regularization term, penalizing large weights to prevent overfitting.
    #Loss function
    box : 1.0                   # default: 7.5, box loss
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

    test_cfg:
    chkpt: best.pt                #chkpt to load for testing
    CM_grouping_vis: True         # whether to group the prediction result in CM for visualization (Recommended for debugging)
    test_param:
        batch: 5                  # batch size, must be >=1
        imgsz: 640
        conf: 0.6
        iou: 0.6
        save_json : True
        plots : True
        verbose : False
        exist_ok : True

### Inference Configuration 

For performing inference on dataset with or without ground truth label. Adjust the setting in the **infer_cfg.yaml** for prediction. 

**infer_cfg.yaml**

    task: classify  #YOLO task, i.e. detect, segment, classify
    dataset : imagenet10
    project : imagenet # name the output project
    name : imagenet # name the output name
    infer_cfg:
    model_chkpt: .\train_out\classify\imagenet10\imagenet\weights\best.engine #relative path for chkpt/deployed model, e.g. .pt, .onnx, .engine .torchscript
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

### Model deployment Configuration 
To deploy a chkpt into a specific format. 
Adjust the **export_cfg.yaml** config.

**export_cfg.yaml**

    export_cfg:
    model_chkpt: .\train_out\classify\imagenet\imagenet\weights\best.pt      #relative path for chkpt/deployed model, e.g. .pt, .onnx, .engine .torchscript
    export_param:
        format: onnx #e.g. onnx, torchscript, engine
        imgsz: 320 # Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions
        half: false # Enables FP16 quantization, reducing model size and potentially speeding up inference
        int8:  false # 	Enables INT8 quantization, highly beneficial for edge deployments
        dynamic: false # Allows dynamic input sizes for ONNX, TensorRT and OpenVINO exports, enhancing flexibility in handling varying image dimensions.
        device: 0 #Specifies the device for exporting: GPU (device=0), CPU (device=cpu)

## Testing and inference result analysis

After performing testing on either the validation or testing of the dataset, there will be multiple folders generated for analysis and visulizing model performance

Below folders are created:

    pred_imgs:          prediction visualization
    pred_labels:        prediction labels (backgrounds are excluded)
    pred_crop:          prediction bounding box crops
    confusion_matrix:   grouping of result in CM
