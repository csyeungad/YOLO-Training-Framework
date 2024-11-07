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
        epochs: 10                  # Number of epochs to train for.
        imgsz: 640                  # Size of input images as integer, imgze : a -> resize longest size to a, while keeping aspect ratio, imgze : (w, h) -> resize img into square
        rect : False                # True: Pad img after resize to square, False: keep original aspect ratio
        batch: 5                    # Specific batch size: batch=16 / Auto mode for % GPU memory utilization (batch= 0.70)
        patience : 20
        lr0 : 0.01
        workers : 1 
        single_cls : False          # (bool) train multi-class data as single-class
        optimizer: auto             # Optimizer to use, choices:[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        save_period : 25            #interval for chkpts save
        plots : True                #save plots and images during train/val
        fraction : 1.0              #train with subset of dataset, 1.0 for full dataset
        exist_ok : True             #If True, allows overwriting of an existing project/name directory
        box : 0.5                   # default: 7.5
        cls : 0.5                   # default: 0.5
        scale : 0.2
        fliplr : 0.0                #flip prob.
        mosaic : 0                  # Combines four training images into one, simulating different scene compositions and object interactions
        close_mosaic : 0
        multi_scale : False         # (bool) Whether to use multiscale during training

    test_cfg:
    chkpt: best.pt                #chkpt to load for testing
    CM_grouping_vis: True         # whether to group the prediction result in CM for visualization (Recommended for debugging)
    test_param:
        batch: 5
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
        save_txt : False
        show : False

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

    pred_imgs:      prediction visualization
    pred_labels:    prediction labels (backgrounds are excluded)
    pred_crop:      prediction bounding box crops

## Model predictions attributes and methods

**model.predict()**

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Object containing detection bounding boxes.
            cls: tensor([3.], device='cuda:0')
            conf: tensor([0.8872], device='cuda:0')
            data: tensor([[ 60.1841, 250.9542, 152.9188, 334.8365,   0.8872,   3.0000]], device='cuda:0')
            id: None
            is_track: False
            orig_shape: (1305, 1528)
            shape: torch.Size([1, 6])
            xywh: tensor([[106.5515, 292.8954,  92.7347,  83.8824]], device='cuda:0')
            xywhn: tensor([[0.0697, 0.2244, 0.0607, 0.0643]], device='cuda:0')
            xyxy: tensor([[ 60.1841, 250.9542, 152.9188, 334.8365]], device='cuda:0')
            xyxyn: tensor([[0.0394, 0.1923, 0.1001, 0.2566]], device='cuda:0')
        masks (Masks | None): Object containing detection masks.
        probs (Probs | None): Object containing class probabilities for classification tasks.
            data: tensor([0.1930, 0.3711, 0.2400, 0.1959], device='cuda:0')
            orig_shape: None
            shape: torch.Size([4])
            top1: 1
            top1conf: tensor(0.3711, device='cuda:0')
            top5: [1, 2, 3, 0]
            top5conf: tensor([0.3711, 0.2400, 0.1959, 0.1930], device='cuda:0')
        keypoints (Keypoints | None): Object containing detected keypoints for each object.
        obb (OBB | None): Object containing oriented bounding boxes.
        speed (Dict[str, float | None]): Dictionary of preprocess, inference, and postprocess speeds.
        names (Dict[int, str]): Dictionary mapping class IDs to class names.
        path (str): Path to the image file.
        _keys (Tuple[str, ...]): Tuple of attribute names for internal use.
    

    
    Methods:
        update: Updates object attributes with new detection results.
        cpu: Returns a copy of the Results object with all tensors on CPU memory.
        numpy: Returns a copy of the Results object with all tensors as numpy arrays.
        cuda: Returns a copy of the Results object with all tensors on GPU memory.
        to: Returns a copy of the Results object with tensors on a specified device and dtype.
        new: Returns a new Results object with the same image, path, and names.
        plot: Plots detection results on an input image, returning an annotated image.
        show: Shows annotated results on screen.
        save: Saves annotated results to file.
        verbose: Returns a log string for each task, detailing detections and classifications.
        save_txt: Saves detection results to a text file.
        save_crop: Saves cropped detection images.
        tojson: Converts detection results to JSON format.
    