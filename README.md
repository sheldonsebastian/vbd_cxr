## README
 
#### Steps to run:

1. Download processed data from .... and download external data from ... and place in root directory as "input_data" folder name.
2. Run requirements.txt to install all required packages
3. For training the models run ... or download trained models from ...
4. For inference run the scripts ...
 
#### Directory Structure:

|Path|Description|
|------------|-----------|
|0_preprocessor|Code to convert DICOM to jpeg and resize images.|
|1_data_split|Code to split the data into train-validation-holdout|
|2_trainer|Code to train classification models and object detection models.|
|3_inference|Code to make predictions using classification, object detection, and ensemble models.|
|4_kaggle_files|Utility files to make Kaggle submissions|
|5_deployment_files|Code related to Flask App|
|common|Utility files for coding|
