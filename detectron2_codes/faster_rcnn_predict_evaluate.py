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
import random
import numpy as np
import torch
from detectron2.utils.logger import setup_logger
from common.detectron2_utils import get_detectron_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

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
SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR")
TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR")
DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
WORKERS = int(os.getenv("NUM_WORKERS"))

# %% --------------------
# dynamic
train_gt_dataframe = MERGED_DIR + "/wbf_merged/90_percent_train/object_detection/95_percent" \
                                  "/80_percent/train_df_0.csv"
val_gt_dataframe = MERGED_DIR + "/wbf_merged/90_percent_train/object_detection/95_percent" \
                                "/20_percent/validation_df_0.csv"

# %% -------------------- SETUP LOGGER
setup_logger(output=DETECTRON2_DIR + "/logs/current", name="faster_rcnn")

# %% --------------------DATASET
# lambda is anonymous function
# train dataset
DatasetCatalog.register("train", lambda: get_detectron_dataset(IMAGE_DIR, train_gt_dataframe))
MetadataCatalog.get("train").set(
    thing_classes=["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                   "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                   "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                   "Pulmonary fibrosis"])

# validation dataset
DatasetCatalog.register("validation", lambda: get_detectron_dataset(IMAGE_DIR, val_gt_dataframe))
MetadataCatalog.get("validation").set(
    thing_classes=["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                   "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                   "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                   "Pulmonary fibrosis"])

# %% --------------------ABC: visualize the dataset
# train_metadata = MetadataCatalog.get("train")
# for record in random.sample(get_detectron_dataset(IMAGE_DIR, train_gt_dataframe), 3):
#     img = cv2.imread(record["file_name"])
#     visualize = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
#     out = visualize.draw_dataset_dict(record)
#     plt.title(record["image_id"])
#     plt.imshow(out.get_image()[:, :, ::-1], cmap="gray")
#     plt.tight_layout()
#     plt.show()

# %% --------------------TRAIN
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = WORKERS
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.OUTPUT_DIR = DETECTRON2_DIR + "/faster_rcnn/current"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
# trainer.train()

# %% --------------------
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                 "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# %% --------------------
evaluator = COCOEvaluator("validation", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "validation")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
