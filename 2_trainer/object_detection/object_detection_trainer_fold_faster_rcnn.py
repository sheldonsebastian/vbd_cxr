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

# %% --------------------START HERE
# https://pytorch.org/vision/stable/models.html#faster-r-cnn
# Faster RCNN default parameters + Adam optimizer
import random
from common.object_detection_models import get_faster_rcnn_model_instance
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.CustomDatasets import VBD_CXR_FASTER_RCNN_Train
from torch.utils import tensorboard
from pathlib import Path
import shutil
from datetime import datetime
import math
from common.utilities import prep_gt_target_for_mAP, prep_pred_for_mAP
from common.mAP_utils import mAP_using_package

# %% --------------------set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and VARIABLES
IMAGE_DIR = os.getenv("IMAGE_DIR")
MERGED_DIR = os.getenv("MERGED_DIR")
SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR")
TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR")

folds = [0, 1, 2, 3, 4]

# to get index from cerberus job id
if "SLURM_ARRAY_TASK_ID" in os.environ:
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Cerberus running for fold: {idx}")
else:
    # if running locally, randomly select any 1 fold
    idx = random.choice(range(0, 5, 1))
    print(f"Locally running for fold: {idx}")

# define the fold
fold = folds[idx]

# %% --------------------TENSORBOARD DIRECTORY INITIALIZATION
train_tensorboard_dir = f"{TENSORBOARD_DIR}/object_detection/{fold}/train"
validation_tensorboard_dir = f"{TENSORBOARD_DIR}/object_detection/{fold}/validation"

# if logs already exist then delete them
train_dirpath = Path(train_tensorboard_dir)
if train_dirpath.exists() and train_dirpath.is_dir():
    shutil.rmtree(train_dirpath)

validation_dirpath = Path(validation_tensorboard_dir)
if validation_dirpath.exists() and validation_dirpath.is_dir():
    shutil.rmtree(validation_dirpath)

# create new tensorboard events directories
train_writer = tensorboard.SummaryWriter(train_tensorboard_dir)
validation_writer = tensorboard.SummaryWriter(validation_tensorboard_dir)

# %% --------------------Data transformations using albumentations
# augmentation of data for training data only
# TODO need to verify augmentations by visualizing it with bounding boxes
# generic_transformer = albumentations.Compose([
#     # augmentation operations
#     albumentations.augmentations.transforms.ColorJitter(brightness=0.5, contrast=0.5,
#                                                         saturation=0.5, hue=0.5,
#                                                         always_apply=False,
#                                                         p=0.5),
#     # horizontal flipping
#     albumentations.augmentations.transforms.HorizontalFlip(p=0.5),
# ])

# %% --------------------DATASET
# NOTE THE DATASET IS GRAY SCALE AND HAS MIN SIDE 1024 AND IS NORMALIZED BY FASTER RCNN
train_data_set = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                           MERGED_DIR + "/wbf_merged/k_fold_splits"
                                                        "/object_detection/train_df_5_folds.csv",
                                           albumentation_transformations=None,
                                           fold=fold)

validation_data_set = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                                MERGED_DIR + "/wbf_merged/k_fold_splits"
                                                             "/object_detection"
                                                             "/validation_df_5_folds.csv",
                                                albumentation_transformations=None,
                                                fold=fold)


# %% --------------------COLLATE FUNCTION required since the image are not of same size
# https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
# https://github.com/pytorch/vision/blob/master/references/detection/utils.py
def collate_fn(batch):
    # https://www.geeksforgeeks.org/zip-in-python/
    # zip(*x) is used to unzip x, where x is iterator
    # thus in the end we will have [(img_id, img_id, ...), (img, img, ...), (target, target, ...)]
    return tuple(zip(*batch))


# %% --------------------DATALOADER
BATCH_SIZE = 8
workers = int(os.getenv("NUM_WORKERS"))

train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=workers,
                                                collate_fn=collate_fn)

validation_data_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=BATCH_SIZE,
                                                     shuffle=False, num_workers=workers,
                                                     collate_fn=collate_fn)

# # %% --------------------
# for i in range(5):
#     images, t = next(iter(train_data_loader))
#     print(images)

# %% --------------------
# define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# %% --------------------MODEL INSTANCE
# 15 = 14 classes (abnormalities) + 1 background class (class=0)
# NOTE:: no findings class is ignored
num_classes = 15

# initializing a pretrained model of Faster RCNN with ResNet50-FPN as Backbone
# NOTE:: FASTER RCNN PyTorch implementation performs normalization based on ImageNet
model = get_faster_rcnn_model_instance(num_classes)

# %% --------------------HYPER-PARAMETERS
LR = 1e-3
EPOCHS = 30

# %% --------------------OPTIMIZER
# freeze the params for pretrained model
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LR)

# %% --------------------move to device
model.to(device)

# %% --------------------TRAINING LOOP
print("Program started")

train_iter = 0
val_iter = 0

lowest_val_loss = 100000

# if directory does not exist then create it
saved_model_dir = Path(f"{SAVED_MODEL_DIR}/object_detection/{fold}/")

if not saved_model_dir.exists():
    os.makedirs(saved_model_dir)

# save model path
saved_model_path = f"{saved_model_dir}/faster_rcnn.pt"

# TRAIN
sum_train_losses_arr = []
mAP_train_arr = []
# FC Layer losses for object detection
train_loss_classifier_arr = []
train_loss_box_reg_arr = []
# RPN Losses for ROI generator
train_loss_objectness_arr = []
train_loss_rpn_box_reg_arr = []

# VALIDATION
sum_validation_losses_arr = []
mAP_validation_arr = []
# FC Layer losses for object detection
validation_loss_classifier_arr = []
validation_loss_box_reg_arr = []
# RPN Losses for ROI generator
validation_loss_objectness_arr = []
validation_loss_rpn_box_reg_arr = []

# start time
start = datetime.now()

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
for epoch in range(EPOCHS):

    print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
    print('-' * 10)

    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    train_true = []

    # [xmin, ymin, xmax, ymax, class_id, confidence]
    train_preds = []

    # running stats for iteration level
    running_sum_train_losses = 0
    running_train_loss_classifier = 0
    running_train_loss_box_reg = 0
    running_train_loss_objectness = 0
    running_train_loss_rpn_box_reg = 0

    # https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    # ----------------------TRAINING DATA --------------
    # iterate through train data in batches
    for train_images, train_targets in train_data_loader:
        # input data as batch
        train_images = list(image.to(device) for image in train_images)

        train_targets_device = []

        for t in train_targets:
            # add targets to train_true for computing mAP
            train_true.extend(prep_gt_target_for_mAP(t))

            train_targets_dict = {}
            # send target to device
            for k, v in t.items():
                train_targets_dict[k] = v.to(device)

            train_targets_device.append(train_targets_dict)
        # ------------------- LOSS --------------------------------------
        # set model to train phase
        model.train()

        # optimizer.zero_grad() is critically important because it resets all weight and bias
        # gradients to 0
        optimizer.zero_grad()

        # forward pass
        # track history in training mode
        with torch.set_grad_enabled(True):
            # make predictions and return all losses (FC Layer + RPN)
            # losses computed over the batch
            loss_dict = model(train_images, train_targets_device)

            # sum all losses
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict)
                sys.exit(1)

            # loss.backward() method uses the back-propagation algorithm to compute all the
            # gradients associated with the weights and biases that are part of the network
            # containing loss_val
            losses.backward()

            # optimizer.step() statement uses the newly computed gradients to update all the
            # weights and biases in the neural network so that computed output values will get
            # closer to the target values
            optimizer.step()

        running_sum_train_losses += losses.item()
        running_train_loss_classifier += loss_dict["loss_classifier"].item()
        running_train_loss_box_reg += loss_dict["loss_box_reg"].item()
        running_train_loss_objectness += loss_dict["loss_objectness"].item()
        running_train_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

        if train_iter % 50 == 0:
            print(f"Training Total Loss for Iteration {train_iter} :: {losses.item()}")

        # ------------------- mAP --------------------------------------
        # compute train mAP
        # set model to eval phase
        model.eval()
        with torch.set_grad_enabled(False):
            # make predictions
            outputs = model(train_images)

            for output in outputs:
                train_preds.extend(prep_pred_for_mAP(output))

        train_iter += 1

    # track mAP at epoch level
    train_mAP = mAP_using_package(train_preds, train_true)
    mAP_train_arr.append(train_mAP)
    train_writer.add_scalar("mAP", train_mAP, global_step=epoch)
    print(f"Train mAP Epoch {epoch}: {train_mAP}")

    # average the losses per epoch
    train_sum_train_losses_agg = running_sum_train_losses / len(train_data_loader.dataset)
    sum_train_losses_arr.append(train_sum_train_losses_agg)
    train_writer.add_scalar("sum_loss", train_sum_train_losses_agg, global_step=epoch)
    print(f"Training Epoch #{epoch} Sum Loss:{train_sum_train_losses_agg}")

    train_loss_classifier_agg = running_train_loss_classifier / len(train_data_loader.dataset)
    train_loss_classifier_arr.append(train_loss_classifier_agg)
    train_writer.add_scalar("loss_classifier", train_loss_classifier_agg, global_step=epoch)

    train_loss_box_reg_agg = running_train_loss_box_reg / len(train_data_loader.dataset)
    train_loss_box_reg_arr.append(train_loss_box_reg_agg)
    train_writer.add_scalar("loss_box_reg", train_loss_box_reg_agg, global_step=epoch)

    train_loss_objectness_agg = running_train_loss_objectness / len(train_data_loader.dataset)
    train_loss_objectness_arr.append(train_loss_objectness_agg)
    train_writer.add_scalar("rpn_loss_objectness", train_loss_objectness_agg, global_step=epoch)

    train_loss_rpn_box_reg_agg = running_train_loss_rpn_box_reg / len(train_data_loader.dataset)
    train_loss_rpn_box_reg_arr.append(train_loss_rpn_box_reg_agg)
    train_writer.add_scalar("rpn_loss_rpn_box_reg", train_loss_rpn_box_reg_agg, global_step=epoch)

    # https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    # ----------------------VALIDATION DATA --------------
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    validation_true = []

    # [xmin, ymin, xmax, ymax, class_id, confidence]
    validation_preds = []

    # running stats for iteration level
    running_sum_validation_losses = 0
    running_validation_loss_classifier = 0
    running_validation_loss_box_reg = 0
    running_validation_loss_objectness = 0
    running_validation_loss_rpn_box_reg = 0

    # iterate through train data in batches
    for validation_images, validation_targets in validation_data_loader:
        # input data as batch
        validation_images = list(image.to(device) for image in validation_images)

        validation_targets_device = []

        for t in validation_targets:
            # add targets to train_true for computing mAP
            validation_true.extend(prep_gt_target_for_mAP(t))

            validation_targets_dict = {}
            # send target to device
            for k, v in t.items():
                validation_targets_dict[k] = v.to(device)

            validation_targets_device.append(validation_targets_dict)
        # ------------------- LOSS --------------------------------------
        # set model to train phase
        model.train()

        # forward pass
        # https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333
        # dont track history thus set grad to False
        with torch.set_grad_enabled(False):
            # make predictions and return all losses (FC Layer + RPN)
            # losses computed over the batch
            loss_dict = model(validation_images, validation_targets_device)

            # sum all losses
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping validation".format(loss_value))
                print(loss_dict)
                sys.exit(1)

        running_sum_validation_losses += losses.item()
        running_validation_loss_classifier += loss_dict["loss_classifier"].item()
        running_validation_loss_box_reg += loss_dict["loss_box_reg"].item()
        running_validation_loss_objectness += loss_dict["loss_objectness"].item()
        running_validation_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

        if val_iter % 50 == 0:
            print(f"Validation Total Loss for Iteration {val_iter} :: {losses.item()}")

        # ------------------- mAP --------------------------------------
        # compute validation mAP
        # set model to eval phase
        model.eval()
        with torch.set_grad_enabled(False):
            # make predictions
            outputs = model(validation_images)

            for output in outputs:
                validation_preds.extend(prep_pred_for_mAP(output))

        val_iter += 1

    # track mAP at epoch level
    val_mAP = mAP_using_package(validation_preds, validation_true)
    mAP_validation_arr.append(val_mAP)
    validation_writer.add_scalar("mAP", val_mAP, global_step=epoch)
    print(f"Validation mAP Epoch {epoch}: {val_mAP}")

    # average the losses per epoch
    validation_sum_train_losses_agg = running_sum_validation_losses / len(
        validation_data_loader.dataset)
    sum_validation_losses_arr.append(validation_sum_train_losses_agg)
    validation_writer.add_scalar("sum_loss", validation_sum_train_losses_agg, global_step=epoch)
    print(f"Validation Epoch #{epoch} Sum Loss:{validation_sum_train_losses_agg}")

    val_loss_classifier_agg = running_validation_loss_classifier / len(
        validation_data_loader.dataset)
    validation_loss_classifier_arr.append(val_loss_classifier_agg)
    validation_writer.add_scalar("loss_classifier", val_loss_classifier_agg, global_step=epoch)

    val_loss_box_reg_agg = running_validation_loss_box_reg / len(validation_data_loader.dataset)
    validation_loss_box_reg_arr.append(val_loss_box_reg_agg)
    validation_writer.add_scalar("loss_box_reg", val_loss_box_reg_agg, global_step=epoch)

    val_loss_objectness_agg = running_validation_loss_objectness / len(
        validation_data_loader.dataset)
    validation_loss_objectness_arr.append(val_loss_objectness_agg)
    validation_writer.add_scalar("rpn_loss_objectness", val_loss_objectness_agg, global_step=epoch)

    val_loss_rpn_box_reg_agg = running_validation_loss_rpn_box_reg / len(
        validation_data_loader.dataset)
    validation_loss_rpn_box_reg_arr.append(val_loss_rpn_box_reg_agg)
    validation_writer.add_scalar("rpn_loss_rpn_box_reg", val_loss_rpn_box_reg_agg,
                                 global_step=epoch)

    # save the best model based on validation loss
    if validation_sum_train_losses_agg < lowest_val_loss:
        lowest_val_loss = validation_sum_train_losses_agg
        # save model state based on lowest val loss per epoch
        torch.save(model.state_dict(), saved_model_path)

# %% --------------------
print("-" * 25)
# print aggregates
# TRAIN
print(f"Train: Average: mAP: {np.mean(mAP_train_arr)}")
print(f"Train: Average: Sum Loss: {np.mean(sum_train_losses_arr)}")
print(f"Train: Average: Loss Classifier: {np.mean(train_loss_classifier_arr)}")
print(f"Train: Average: Loss Box Reg: {np.mean(train_loss_box_reg_arr)}")
print(f"Train: Average: RPN Loss Objectness: {np.mean(train_loss_objectness_arr)}")
print(f"Train: Average: Loss RPN Box Reg: {np.mean(train_loss_rpn_box_reg_arr)}")

# validation
print(f"Validation: Average: mAP: {np.mean(mAP_validation_arr)}")
print(f"Validation: Average: Sum Loss: {np.mean(sum_validation_losses_arr)}")
print(f"Validation: Average: Loss Classifier: {np.mean(validation_loss_classifier_arr)}")
print(f"Validation: Average: Loss Box Reg: {np.mean(validation_loss_box_reg_arr)}")
print(f"Validation: Average: RPN Loss Objectness: {np.mean(validation_loss_objectness_arr)}")
print(f"Validation: Average: Loss RPN Box Reg: {np.mean(validation_loss_rpn_box_reg_arr)}")
