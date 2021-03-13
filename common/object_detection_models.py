# %% --------------------
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


# %% --------------------
# FASTER RCNN
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn
# https://pytorch.org/vision/stable/models.html#faster-r-cnn
def get_faster_rcnn_model_instance(num_classes=15, min_size=512, pretrained=True):
    # https://discuss.pytorch.org/t/faster-mask-rcnn-rpn-custom-anchorgenerator/69962/2
    # anchor_generator = AnchorGenerator(
    #     sizes=((16,), (32,), (64,), (128,), (256,)),
    #     aspect_ratios=tuple([(0.25, 0.5, 1.0, 1.5, 2.0) for _ in range(5)]))

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 min_size=min_size)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# %% --------------------
# RETINA NET

# %% --------------------
# MASK RCNN
