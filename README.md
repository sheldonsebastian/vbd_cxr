# VinBigData Chest X-ray Abnormality Detection
 
 Steps to reproduce results:
 .....
 
 Directory Structure:
 |Path|Description|
 |------------|-----------|
 |....|.....|
 
- 0_preprocessor ==> DICOM to jpeg
- 1_merger ==> Bounding Box Fusion using WBF
- 2_trainer
    - model_trainer.py ==> Has Everything for Faster RCNN + Dataset + Dataloader + Subset(Train and Validation splits)

- 3_validation_data
   - validation predictor.py ==> loads saved model and loads the validation data. Makes prediction and stores in validation_predictions.csv
   - validation visualizer.py ==> visualizes predicted vs ground truth values (WBF + NMS)
   - validation mAP.py ==> Uses COCO api to copmute mAP for 0.4 IoU
   - validation upscaling.py ==> convert the predicted bounding boxes for 1024 image to original dimension
 
- 4_testing_data
   - test predictor.py ==> loads saved model and loads the testing data. Makes prediction and stores in test_predictions.csv
   - test visualizer.py ==> visualizes the predictions
   - test merge bb ==> merges predictions using NMS or WBF. Also contains logic for No findings class.
   - test upscaling.py ==> validation upscaling.py
   - kaggle.py ==> creates kaggle output to submit
