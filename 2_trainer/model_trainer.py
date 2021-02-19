# %% --------------------
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings("ignore")

# %% --------------------
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# %% --------------------
# directories
# image dir
IMAGE_DIR = "/groups/dats6501/ssebastian94/transformed_data/train"

BB_FILE = "/groups/dats6501/ssebastian94/merged_bounding_boxes/abnormalities_bb.csv"

# using fused train data with IoU threshold of 0.6 only abnormalities
train_bb = pd.read_csv(BB_FILE)

# saved model directory
saved_model_path = "/groups/dats6501/ssebastian94/saved_model.pt"

# train indices
train_indices = "/groups/dats6501/ssebastian94/Training Indices.txt"

# validation indices
validation_indices ="/groups/dats6501/ssebastian94/Validation Indices.txt"

# %% --------------------
# 0 = disable multiprocessing
workers = 4

# %% --------------------
print(train_bb.head())

# %% --------------------

label2color = {
    0: ("Aortic enlargement", "#2a52be"),
    1: ("Atelectasis", "#ffa812"),
    2: ("Calcification", "#ff8243w"),
    3: ("Cardiomegaly", "#4682b4"),
    4: ("Consolidation", "#ddadaf"),
    5: ("ILD", "#a3c1ad"),
    6: ("Infiltration", "#008000"),
    7: ("Lung Opacity", "#004953"),
    8: ("Nodule/Mass", "#e3a857"),
    9: ("Other lesion", "#dda0dd"),
    10: ("Pleural effusion", "#e6e8fa"),
    11: ("Pleural thickening", "#800020"),
    12: ("Pneumothorax", "#918151"),
    13: ("Pulmonary fibrosis", "#e75480")
}


# %% --------------------

# dataset used for training
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
        '''getitem should return image, target dictionary {boxes[x0,y0,x1,y1], labels, image_id, area, iscrowd}'''
        image_id = self.image_ids[index]
        image_data = self.data[self.data["image_id"] == image_id]

        # image
        # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/2 ==> Convert greyscale to RGB
        image = Image.open(self.base_dir + "/" + image_id + ".jpeg").convert('RGB')

        # boxes
        boxes = image_data[['x_min', 'y_min', 'x_max', 'y_max']].values
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # convert everything to tensor
        boxes = torch.FloatTensor(boxes)
        area = torch.FloatTensor(area)

        # instances with iscrowd=True will be ignored during evaluation.
        # here we set all to False since we are using zeros
        iscrowd = torch.zeros((image_data.shape[0]), dtype=torch.int64)

        labels = torch.as_tensor(image_data["class_id"].values, dtype=torch.int64)

        # dictionary as required by Faster RCNN
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([index]), "area": area,
                  "iscrowd": iscrowd}

        # transform image to tensor
        image = T.ToTensor()(image)

        return image, target

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
dataset = VinBigDataCXR(IMAGE_DIR, BB_FILE)
dataset_validation = VinBigDataCXR(IMAGE_DIR, BB_FILE)

# %% --------------------
# train_dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=True)

# for i in range(5):
#    images, t = next(iter(train_dataloader))
#    print(images)
#    print(images.shape)
#    print(t)

# %% --------------------
validation_size = round(0.1 * len(dataset))

# %% --------------------

# split the dataset in train and validation set 90%-10%
indices = torch.randperm(len(dataset)).tolist()

dataset = torch.utils.data.Subset(dataset, indices[:-validation_size])
dataset_validation = torch.utils.data.Subset(dataset_validation, indices[-validation_size:])

# %% --------------------
print(len(dataset))

# %% --------------------
print(len(dataset_validation))

# %% --------------------
# https://stackoverflow.com/a/33686762
with open(train_indices, "w") as output:
    output.write(str(indices[:-validation_size]))

with open(validation_indices, "w") as output:
    output.write(str(indices[-validation_size:]))


# %% --------------------

# https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
# https://github.com/pytorch/vision/blob/master/references/detection/utils.py
def collate_fn(batch):
    # https://www.geeksforgeeks.org/zip-in-python/
    # zip(*x) is used to unzip x, where x is iterator
    return tuple(zip(*batch))


# %% --------------------
BATCH_SIZE_TRAIN = 5

# define training data loaders using the subset defined earlier
train_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=workers,
    collate_fn=collate_fn)

#
### %% --------------------
#for images, targets in train_data_loader:
#    print(images)
#    print(targets)

# %% --------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# %% --------------------

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

model = get_model_instance()

# %% --------------------
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# %% --------------------
num_epochs = 10

# %% --------------------
# move model to device
model.to(device)

# %% --------------------
print("Training started")
# start time
start = datetime.now()

lowest_loss_value = 100
losses_arr = []
itr = 1
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://github.com/pytorch/vision/blob/master/references/detection/engine.py
for epoch in range(num_epochs):
    
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # data_loader is training data
    for images, targets in train_data_loader:
        print(itr)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        losses_arr.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1

    print(f"Epoch #{epoch} loss: {loss_value}")

    # code to avoid overfitting
    # if loss is less than lowest loss, only then save the model
    if loss_value < lowest_loss_value:
        torch.save(model.state_dict(), saved_model_path)

        # update lowest_loss_value
        lowest_loss_value = loss_value

print("Training Complete")
print("End time:" + str(datetime.now() - start))

print(f"Average Loss:{np.mean(losses_arr)}")

# %% --------------------
