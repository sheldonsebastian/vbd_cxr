# %% --------------------
import os
import sys

from dotenv import load_dotenv

# %% --------------------
# local
# env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
#           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# %% --------------------
# DIRECTORIES
SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_DIR") + "/saved_model_20210212.pt"
VALIDATION_INDICES = os.getenv("VALIDATION_INDICES")
IMAGE_DIR = os.getenv("IMAGE_DIR")
BB_FILE = os.getenv("BB_FILE")
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------
# Reference:
# https://www.kaggle.com/pestipeti/vinbigdata-fasterrcnn-pytorch-inference?scriptVersionId=50935253
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from common.utilities import read_text_literal


# %% --------------------
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# %% --------------------
# workers for dataloader
workers = 4


# %% --------------------
# initialize model
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model_instance():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 15  # 14 classes (abnormalities) + background (class=0)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# %% --------------------
# get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Torch is using following device: " + str(device))

# %% --------------------
# load saved model to appropriate device
model = get_model_instance()
model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=torch.device(device)))

# %% --------------------
# set model to eval mode, to not disturb the weights
model.eval()

# %% --------------------
# send model to device
model = model.to(device)


# %% --------------------
# dataset used for validation
# https://pytorch.org/docs/stable/data.html#map-style-datasets
class VinBigDataCXR(Dataset):

    def __init__(self, image_dir, annotation_file_path):
        super().__init__()
        self.base_dir = image_dir
        self.data = pd.read_csv(annotation_file_path)

        # Change class_id of BB, since FasterRCNN assumes class_id==0 is background.
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        self.data["class_id"] = self.data["class_id"] + 1

        # sorted the image_ids
        self.image_ids = sorted(self.data["image_id"].unique())

    def __getitem__(self, index):
        '''getitem should return image, image_id'''
        image_id = self.image_ids[index]

        # image
        # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/2 ==> Convert greyscale to RGB
        image = Image.open(self.base_dir + "/" + image_id + ".jpeg").convert('RGB')

        # transform image to tensor
        image = T.ToTensor()(image)

        # return image_id so we can use it for validation
        return image, image_id

    def __len__(self):
        return len(self.image_ids)

    def __get_height_and_width__(self, index):
        # https://discuss.pytorch.org/t/datasets-aspect-ratio-grouping-get-get-height-and-width/62640/2
        ''' if you want to use aspect ratio grouping during training (so that each batch only
        contains images with similar aspect ratio), then it is recommended to also implement
        a get_height_and_width method, which returns the height and the width of the image.'''

        image_id = self.image_ids[index]
        image = Image.open(self.base_dir + "/" + image_id + ".jpeg")
        width, height = image.size

        return height, width


# %% --------------------
# validation dataset
dataset_validation = VinBigDataCXR(IMAGE_DIR, BB_FILE)

# %% --------------------
# validation indices
validation_indices = read_text_literal(VALIDATION_INDICES)

# %% --------------------
# subset the data using validation indices
dataset_validation = torch.utils.data.Subset(dataset_validation, validation_indices)


# %% --------------------
# https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
# https://github.com/pytorch/vision/blob/master/references/detection/utils.py
def collate_fn(batch):
    # https://www.geeksforgeeks.org/zip-in-python/
    # zip(*x) is used to unzip x, where x is iterator
    return tuple(zip(*batch))


# create dataloader
BATCH_SIZE_VALIDATION = 5

# define training data loaders using the subset defined earlier
validation_data_loader = torch.utils.data.DataLoader(
    dataset_validation, batch_size=BATCH_SIZE_VALIDATION, shuffle=False, num_workers=workers,
    collate_fn=collate_fn)

# %% --------------------
# make predictions
print("Validation predictions started")
# start time
start = datetime.now()

# arrays
image_id = []
x_min = []
y_min = []
x_max = []
y_max = []
label_arr = []
confidence_score = []

with torch.no_grad():
    for images, image_ids in validation_data_loader:
        # iterate through images and send to device
        images_device = list(image.to(device) for image in images)

        # output is list of dictionary [{boxes:tensor([[xmin, ymin, xmax, ymax], [...]],
        # device=cuda), labels:tensor([15, 11, ...], device=cuda), scores:tensor([0.81, 0.92,
        # ...], device=cuda)},{...}]
        outputs = model(images_device)
        print(image_ids)
        print(outputs)

        for img_id, output in zip(image_ids, outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                image_id.append(img_id)
                x_min.append(box[0])
                y_min.append(box[1])
                x_max.append(box[2])
                y_max.append(box[3])
                label_arr.append(label)
                confidence_score.append(score)

print("Predictions Complete")
print("End time:" + str(datetime.now() - start))

val_predictions = pd.DataFrame({"image_id": image_id,
                                "x_min": x_min,
                                "y_min": y_min,
                                "x_max": x_max,
                                "y_max": y_max,
                                "label": label_arr,
                                "confidence_score": confidence_score})

# %% --------------------
# write csv file
val_predictions.to_csv(VALIDATION_PREDICTION_DIR + "/validation_predictions.csv", index=False)
