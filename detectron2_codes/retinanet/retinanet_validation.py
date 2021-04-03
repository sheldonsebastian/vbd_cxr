# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
# env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
#            "Workspace/vbd_cxr/6_environment_files/local.env "

# cerberus
env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------IMPORTS
# https://www.kaggle.com/corochann/vinbigdata-detectron2-train
import cv2
from detectron2.engine import DefaultPredictor
import random
import numpy as np
import torch
from detectron2.utils.logger import setup_logger
from common.detectron2_utils import get_test_detectron_dataset, predict_batch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2 import model_zoo
from common.detectron_config_manager import Flags
import pandas as pd

# %% --------------------set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and VARIABLES
IMAGE_DIR = os.getenv("IMAGE_DIR")
# MERGED_DIR contains GT dataframes
MERGED_DIR = os.getenv("MERGED_DIR")
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
WORKERS = int(os.getenv("NUM_WORKERS"))

# %% --------------------READ DATA
# DYNAMIC
holdout_gt_dataframe = MERGED_DIR + f"/512/unmerged/10_percent_holdout/holdout_df.csv"
flag_path = DETECTRON2_DIR + "/retinanet/configurations/v2.yaml"
output_dir = DETECTRON2_DIR + f"/retinanet/holdout/current"
model_dir = DETECTRON2_DIR + f"/retinanet/train/final"

# %% --------------------READ FLAGS
flag = Flags().load_yaml(flag_path)

# %% -------------------- SETUP LOGGER
setup_logger(output=output_dir)

# %% --------------------REGISTER DATASETs and METADATA
thing_classes = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                 "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                 "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                 "Pulmonary fibrosis"]

# lambda is anonymous function
# holdout dataset w/o annotations
DatasetCatalog.register("holdout",
                        lambda: get_test_detectron_dataset(IMAGE_DIR, holdout_gt_dataframe))
MetadataCatalog.get("holdout").set(thing_classes=thing_classes)

# %% --------------------CONFIGURATIONS
cfg = get_cfg()

# no augmentation for holdout dataset
cfg.aug_kwargs = CN(flag.get("aug_kwargs"))

# update output directory
cfg.OUTPUT_DIR = output_dir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# %% --------------------MODEL CONFIGURATION
config_name = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_name))
# use saved model weights
cfg.MODEL.WEIGHTS = model_dir + "/model_final.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# update model anchor sizes and aspect ratio
# https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/220295
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[2, 4, 8, 16, 32, 64, 128, 256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 2.5]]

# update the number of classes
cfg.MODEL.RETINANET.NUM_CLASSES = len(thing_classes)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

# model prediction threshold
print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05

# set a custom testing threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)

# %% --------------------OPTIMIZER CONFIGURATION
batch_size = int(flag.get("batch_size"))
# define batch size
cfg.SOLVER.IMS_PER_BATCH = batch_size

# %% --------------------DATASET CONFIGURATION
# define testing dataset
cfg.DATASETS.TEST = ("holdout",)

# define num worker for dataloader
cfg.DATALOADER.NUM_WORKERS = WORKERS

# %% --------------------INFERENCE
# this will create model and load weights and use cfg.DATASETS.TEST[0]
predictor = DefaultPredictor(cfg)
holdout_dataset = DatasetCatalog.get("holdout")

# %% --------------------
outputs_arr = []

# iterate for MAX ITERATION
for i in range(int(np.ceil(len(holdout_dataset) / batch_size))):
    # get image indices
    inds = list(range(batch_size * i, min(batch_size * (i + 1), len(holdout_dataset))))
    dataset_dicts_batch = [holdout_dataset[i] for i in inds]

    # read the indices of the image
    im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]

    # make predictions for batch of input images
    outputs_list = predict_batch(predictor, im_list)

    # format the batch of outputs
    for idx, output in zip(inds, outputs_list):
        # get image id
        img_id = holdout_dataset[idx]["image_id"]

        fields = output["instances"].get_fields()

        # bb coordinates + labels + score
        for bb, label, score in zip(fields["pred_boxes"], fields["pred_classes"], fields["scores"]):
            outputs_arr.append({
                # image_id,x_min,y_min,x_max,y_max,label,confidence_score
                "image_id": img_id,
                "x_min": bb[0].item(),
                "y_min": bb[1].item(),
                "x_max": bb[2].item(),
                "y_max": bb[3].item(),
                "label": label.item(),
                "confidence_score": score.item()
            })

# convert outputs to dataframe
results = pd.DataFrame(outputs_arr,
                       columns=["image_id", "x_min", "y_min", "x_max", "y_max", "label",
                                "confidence_score"])

print("Finished Inference")
# %% --------------------
# NOTE THERE CAN BE IMAGE IDS WHICH ARE NOT ADDED IN PREDICTION CSV, POST PROCESSING NEEDED
# store dataframe as csv
results.to_csv(output_dir + "/holdout.csv", index=False)
