## README

### Final Report:

https://sheldonsebastian.github.io/vbd_cxr/

### Directory Structure:

|Path|Description|
|------------|-----------|
|0_preprocessor|Code to convert DICOM to png and resize images.|
|1_eda|Code to perform EDA|
|2_data_split|Code to split the data into train-validation-holdout|
|3_trainer|Code to train classification models and object detection models.|
|4_saved_models|Saved models are stored here. Download trained models from [here](https://www.kaggle.com/sheldonsebastian/vbd-cxr-files)|
|5_inference_on_holdout_10_percent|Code to make predictions using classification, object detection, and ensemble models.|
|6_inference_on_kaggle_test_files|Utility files to make Kaggle submissions|
|7_deployment_files|Code related to Flask App|
|common|Utility files for making coding easier|
|archived|Contains Proof of Concepts and miscellaneous files for experimentation purposes|
|docs| files related to GitHub website|
|input_data| folder in which input data will be placed|

### Steps to replicate project:

1. Download processed data from [here](https://www.kaggle.com/awsaf49/vinbigdata-512-image-dataset) and download external data from [here](https://www.kaggle.com/sheldonsebastian/external-cxr-dataset) and place in root directory as "input_data" folder name.
2. To create data train-holdout split for classification and object detection models, run all scripts in 2_data_split in the order they appear.
3. Download trained models from [here](https://www.kaggle.com/sheldonsebastian/vbd-cxr-files) or run all the scripts in 3_trainer.
4. To make inference on holdout dataset using:
   
    a. classification models run all scripts in 5_inference_on_holdout_10_percent/1_classification_models folder.
   
    b. object detection model run all scripts in 5_inference_on_holdout_10_percent/2_object_detection_models folder.
   
    c. ensemble model run all scripts in 5_inference_on_holdout_10_percent/3_ensemble folder.
   
4. To make inference on kaggle test dataset run all scripts in 6_inference_on_kaggle_test_files folder.

### Additional Packages required:

1. [albumentations](https://albumentations.ai/)
2. [pytorch](https://pytorch.org/)
3. [detectron2](https://github.com/facebookresearch/detectron2)
4. [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
5. [Mean-Average-Precision-for-Boxes](https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes)
6. [YOLOv5](https://github.com/ultralytics/yolov5)