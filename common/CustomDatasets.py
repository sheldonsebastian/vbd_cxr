# %% --------------------
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


# %% --------------------
# dataset used for training
# https://pytorch.org/docs/stable/data.html#map-style-datasets
class VBD_CXR_2_Class_Train(Dataset):

    def __init__(self, image_dir, annotation_file_path, majority_transformations, fold=None,
                 minority_class=None, minority_transformations=None, target_column="class_id"):
        """
        :image_dir: The path where all the images are present
        :annotation_file_path: csv file which contains image_id and label. 1 row = 1 image,
        not bounding box data, so drop duplicates
        :fold: specifies the fold to use. If fold=None, then entire dataset will be used
        :majority_transformations: albumentation transformations to perform on majority class
        :minority_class: This is list in which we pass the minority classes
        :minority_transformations: albumentation transformations to perform on minority class
        :target_column: the target column in the annotation_file_path pandas dataframe
        """
        super().__init__()
        self.base_dir = image_dir

        self.data = pd.read_csv(annotation_file_path)

        # subset data based on fold
        if fold is not None:
            self.data = self.data[self.data["fold"] == fold]

        # sorted the image_ids
        self.image_ids = sorted(self.data["image_id"].unique())

        # majority transformations
        self.majority_transformations = majority_transformations

        # minority class
        self.minority_class = minority_class

        # minority transformations
        self.minority_transformations = minority_transformations

        # target classes as list used to handle class imbalance
        self.targets = self.data[target_column].tolist()

    def __getitem__(self, index):
        """getitem should return image and label"""

        image_id = self.image_ids[index]
        target = self.data[self.data["image_id"] == image_id]["class_id"].values[0]

        # handle transformations for majority and minority classes
        if (self.minority_class is not None) and (target in self.minority_class):
            transformations = self.minority_transformations
        else:
            transformations = self.majority_transformations

        # convert target to Tensor
        target = torch.as_tensor(target, dtype=torch.float)

        # image https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/2 ==> Convert
        # greyscale to RGB
        image = Image.open(self.base_dir + "/" + image_id + ".jpeg").convert('RGB')

        # convert image to numpy array
        image = np.asarray(image)

        # apply transformations
        transformed = transformations(image=image)

        # convert image to tensor
        image = T.ToTensor()(transformed["image"])

        return image_id, image, target

    def __len__(self):
        return len(self.image_ids)


# %% --------------------
# dataset used for faster rcnn training
# https://pytorch.org/docs/stable/data.html#map-style-datasets
class VBD_CXR_FASTER_RCNN_Train(Dataset):

    def __init__(self, image_dir, annotation_file_path, albumentation_transformations, fold=None):
        super().__init__()

        self.base_dir = image_dir

        self.data = pd.read_csv(annotation_file_path)
        # subset data based on fold
        if fold is not None:
            self.data = self.data[self.data["fold"] == fold]

        # TODO DELETE THIS
        self.data = self.data.head(10)

        # sorted the image_ids
        self.image_ids = sorted(self.data["image_id"].unique())

        # albumentation transformations
        self.albumentation_transformations = albumentation_transformations

        # Change class_id of BB, since FasterRCNN assumes class_id==0 is background.
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        self.data["class_id"] = self.data["class_id"] + 1

    def __getitem__(self, index):
        """getitem should return image, target dictionary {boxes[x0,y0,x1,y1], labels, image_id,
        area, iscrowd} """
        image_id = self.image_ids[index]
        image_data = self.data[self.data["image_id"] == image_id]

        # read the image as grayscale
        image = Image.open(self.base_dir + "/" + image_id + ".jpeg")

        # boxes
        boxes = image_data[['x_min', 'y_min', 'x_max', 'y_max']].values

        # TODO need to implement and verify data augmentation using albumentations
        # # add albumentation tranformations
        # if self.albumentation_transformations is not None:
        #     transformed = self.albumentation_transformations(image=np.array(image), bboxes=boxes)
        #     image = transformed["image"]
        #     boxes = transformed["bboxes"]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # convert everything to tensor
        boxes = torch.FloatTensor(boxes)
        area = torch.FloatTensor(area)

        # instances with iscrowd=True will be ignored during evaluation.
        # here we set all to False since we are using zeros
        iscrowd = torch.zeros((image_data.shape[0]), dtype=torch.int64)

        # convert target to tensor
        labels = torch.as_tensor(image_data["class_id"].values, dtype=torch.int64)

        # dictionary as required by Faster RCNN
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([index]),
                  "area": area, "iscrowd": iscrowd}

        # transform image to tensor
        image = T.ToTensor()(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)

    def __get_height_and_width__(self, index):
        # https://discuss.pytorch.org/t/datasets-aspect-ratio-grouping-get-get-height-and-width/62640/2
        """ if you want to use aspect ratio grouping during training (so that each batch only
        contains images with similar aspect ratio), then it is recommended to also implement
        a get_height_and_width method, which returns the height and the width of the image."""

        image_id = self.image_ids[index]
        image = Image.open(self.base_dir + "/" + image_id + ".jpeg")
        width, height = image.size

        return height, width

    def get_image_id_using_index(self, index):
        """
        :index: get index from target dictionary from get_item function
        """
        image_id = self.image_ids[index]
        return image_id
