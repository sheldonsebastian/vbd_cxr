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

import torch
from detectron2.utils.logger import setup_logger
from common.detectron2_utils import get_train_detectron_dataset, convert_epoch_to_max_iter, \
    CustomTrainLoop
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2 import model_zoo
from common.detectron_config_manager import Flags

# %% --------------------set seeds
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and VARIABLES
IMAGE_DIR = os.getenv("IMAGE_DIR")
# MERGED_DIR contains GT dataframes
MERGED_DIR = os.getenv("MERGED_DIR")
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
WORKERS = int(os.getenv("NUM_WORKERS"))
EXTERNAL_DIR = os.getenv("EXTERNAL_DIR")

# %% --------------------
# DYNAMIC
train_gt_dataframe = MERGED_DIR + f"/512/unmerged/90_percent_train/object_detection/90_percent" \
                                  f"/train_df.csv"
val_gt_dataframe = MERGED_DIR + f"/512/unmerged/90_percent_train/object_detection/10_percent" \
                                f"/holdout_df.csv"
external_gt_dataframe = EXTERNAL_DIR + "/transformed_train.csv"
flag_path = DETECTRON2_DIR + "/retinanet/configurations/v2.yaml"
output_dir = DETECTRON2_DIR + f"/retinanet/train/final/"

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
# train dataset
DatasetCatalog.register("train", lambda: get_train_detectron_dataset(IMAGE_DIR, train_gt_dataframe,
                                                                     EXTERNAL_DIR,
                                                                     external_gt_dataframe))
MetadataCatalog.get("train").set(thing_classes=thing_classes)

# validation dataset
DatasetCatalog.register("validation",
                        lambda: get_train_detectron_dataset(IMAGE_DIR, val_gt_dataframe))
MetadataCatalog.get("validation").set(thing_classes=thing_classes)

# %% --------------------CONFIGURATIONS
cfg = get_cfg()

# add augmentation dictionary to configuration
cfg.aug_kwargs = CN(flag.get("aug_kwargs"))

# update output directory
cfg.OUTPUT_DIR = output_dir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# %% --------------------MODEL CONFIGURATION
config_name = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_name))
# use pretrained model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# update model anchor sizes and aspect ratio
# https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/220295
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[2, 4, 8, 16, 32, 64, 128, 256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 2.5]]

# update the number of classes
cfg.MODEL.RETINANET.NUM_CLASSES = len(thing_classes)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

# update the minimum and maximum size of image to be used for training
cfg.INPUT.MIN_SIZE_TRAIN = (512,)
cfg.INPUT.MAX_SIZE_TRAIN = (512,)

# %% --------------------OPTIMIZER CONFIGURATION
# define batch size
cfg.SOLVER.IMS_PER_BATCH = flag.get("batch_size")
# LR scheduler name
cfg.SOLVER.LR_SCHEDULER_NAME = flag.get("lr_scheduler_name")
# LR
cfg.SOLVER.BASE_LR = flag.get("base_lr")

# convert epochs to iterations
dataset_length = len(DatasetCatalog.get("train"))
max_iterations = int(convert_epoch_to_max_iter(flag.get("epoch"), flag.get("batch_size"),
                                               dataset_length))
print("MAX ITERATION for Training:" + str(max_iterations))
cfg.SOLVER.MAX_ITER = max_iterations

# define how often to save the model, convert epoch to iterations
# Small value=Frequent save need a lot of storage.
validation_iteration = int(convert_epoch_to_max_iter(flag.get("eval_period_epoch"),
                                                     flag.get("batch_size"), dataset_length))
print("Validation Iteration Period:" + str(validation_iteration))
cfg.SOLVER.CHECKPOINT_PERIOD = validation_iteration

# %% --------------------DATASET CONFIGURATION
# define training dataset
cfg.DATASETS.TRAIN = ("train",)

# define validation dataset
cfg.DATASETS.TEST = ("validation",)

# define after how long the validation dataset should be evaluated
cfg.TEST.EVAL_PERIOD = validation_iteration

# define num worker for dataloader
cfg.DATALOADER.NUM_WORKERS = WORKERS

cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
# anything under 1000 frequency will be repeated more often
cfg.DATALOADER.REPEAT_THRESHOLD = 1000

# %% --------------------EVALUATION PARAMETER
cfg.mAP_conf_thr = 0.10

# %% --------------------TRAINING
# define training loop
trainer = CustomTrainLoop(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
