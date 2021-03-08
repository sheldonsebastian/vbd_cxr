# %% --------------------
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# %% --------------------
# FASTER RCNN
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn
# https://pytorch.org/vision/stable/models.html#faster-r-cnn
def get_faster_rcnn_model_instance(num_classes=15):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# %% --------------------
# RETINA NET

# %% --------------------
# MASK RCNN
