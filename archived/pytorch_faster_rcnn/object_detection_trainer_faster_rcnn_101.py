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
# Faster RCNN
import albumentations
import random
from common.object_detection_models import get_faster_rcnn_101_model_instance
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.CustomDatasets import VBD_CXR_FASTER_RCNN_Train
from torch.utils import tensorboard
from pathlib import Path
import shutil
from common.utilities import extract_image_id_from_batch_using_dataset
from datetime import datetime
from common.mAP_utils import ZFTurbo_MAP_TRAINING, get_id_to_label_mAP
import pandas as pd
import math
from common.CustomErrors import TrainError, ValidationError
import traceback

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

# %% --------------------TENSORBOARD DIRECTORY INITIALIZATION
train_tensorboard_dir = f"{TENSORBOARD_DIR}/object_detection/default/train"
validation_tensorboard_dir = f"{TENSORBOARD_DIR}/object_detection/default/validation"

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
augmentor = albumentations.Compose(
    [
        # augmentation operations
        albumentations.augmentations.transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                            saturation=0.3, hue=0.3,
                                                            always_apply=False,
                                                            p=0.4),
        albumentations.augmentations.transforms.GlassBlur(p=0.2),
        albumentations.augmentations.transforms.GaussNoise(p=0.2),
        albumentations.augmentations.transforms.RandomGamma(p=0.2),

        # horizontal flipping
        albumentations.augmentations.transforms.HorizontalFlip(p=0.4)
    ],
    bbox_params=albumentations.BboxParams(format='pascal_voc')
)

# %% --------------------DATASET
# NOTE THE DATASET IS GRAY SCALE AND HAS MIN SIDE 512 AND IS NORMALIZED BY FASTER RCNN
train_data_set = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                           MERGED_DIR + "/wbf_merged"
                                                        "/object_detection/train_df_80.csv",
                                           albumentation_transformations=augmentor)

train_data_set_mAP = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                               MERGED_DIR + "/wbf_merged"
                                                            "/object_detection/train_df_80.csv",
                                               albumentation_transformations=None)

validation_data_set = VBD_CXR_FASTER_RCNN_Train(IMAGE_DIR,
                                                MERGED_DIR + "/wbf_merged"
                                                             "/object_detection"
                                                             "/val_df_20.csv",
                                                albumentation_transformations=None)


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

# when computing mAP we don't want to perform augmentations
train_data_loader_mAP = torch.utils.data.DataLoader(train_data_set_mAP, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=workers,
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

# initializing a model of Faster RCNN with ResNet101-FPN as Backbone
# NOTE:: FASTER RCNN PyTorch implementation performs normalization based on ImageNet
# NOTE:: FOR RESNET101 there is no pretrained models, thus we need to train longer and there
# is chance of overfitting for training data
model = get_faster_rcnn_101_model_instance(num_classes, 512)

# %% --------------------HYPER-PARAMETERS
LR = 0.005
EPOCHS = 100

# %% --------------------
if EPOCHS // 10 == 0:
    save_step_size = 1
else:
    save_step_size = (EPOCHS // 10)

# %% --------------------OPTIMIZER
# freeze the params for pretrained model
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params=params, momentum=0.9, lr=LR, weight_decay=0.0005)

# %% --------------------LR reduce on plateau
# https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, 'min')

# %% --------------------move to device
model.to(device)

# %% --------------------
train_original_dimension = pd.read_csv(MERGED_DIR + "/wbf_merged/object_detection/train_df_80.csv")
validation_original_dimension = pd.read_csv(
    MERGED_DIR + "/wbf_merged/object_detection/val_df_20.csv")

# %% --------------------TRAINING LOOP
print("Program started")

train_iter = 0
val_iter = 0

# if directory does not exist then create it
saved_model_dir = Path(f"{SAVED_MODEL_DIR}/object_detection/default")

if not saved_model_dir.exists():
    os.makedirs(saved_model_dir)

# TRAIN
sum_train_losses_arr = []
# FC Layer losses for object detection
train_loss_classifier_arr = []
train_loss_box_reg_arr = []
# RPN Losses for ROI generator
train_loss_objectness_arr = []
train_loss_rpn_box_reg_arr = []

# VALIDATION
sum_validation_losses_arr = []
# FC Layer losses for object detection
validation_loss_classifier_arr = []
validation_loss_box_reg_arr = []
# RPN Losses for ROI generator
validation_loss_objectness_arr = []
validation_loss_rpn_box_reg_arr = []

# start time
start = datetime.now()

model_save_counter = 0

try:
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    for epoch in range(EPOCHS):

        print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
        print('-' * 10)

        # https://github.com/pytorch/vision/blob/master/references/detection/engine.py
        # -----------------------COMPUTING LOSSES::TRAINING DATA------------------------------------

        # running stats for iteration level
        running_sum_train_losses = 0
        running_train_loss_classifier = 0
        running_train_loss_box_reg = 0
        running_train_loss_objectness = 0
        running_train_loss_rpn_box_reg = 0

        # iterate through train data in batches
        try:
            for train_images, train_targets in train_data_loader:

                # input data as batch
                train_images = list(image.to(device) for image in train_images)
                train_targets_device = [{k: v.to(device) for k, v in t.items()} for t in
                                        train_targets]

                # set model to train phase
                model.train()

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

                    # optimizer.zero_grad() is critically important because it resets all weight and
                    # bias gradients to 0
                    optimizer.zero_grad()

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

                train_iter += 1

        except Exception as e:
            image_ids_error = extract_image_id_from_batch_using_dataset(train_targets,
                                                                        train_data_loader.dataset)
            raise TrainError(image_ids_error, traceback.format_exc())

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
        train_writer.add_scalar("rpn_loss_rpn_box_reg", train_loss_rpn_box_reg_agg,
                                global_step=epoch)

        # https://github.com/pytorch/vision/blob/master/references/detection/engine.py
        # ---------------------COMPUTING LOSSES::VALIDATION DATA------------------------------------

        # running stats for iteration level
        running_sum_validation_losses = 0
        running_validation_loss_classifier = 0
        running_validation_loss_box_reg = 0
        running_validation_loss_objectness = 0
        running_validation_loss_rpn_box_reg = 0

        # iterate through train data in batches
        try:
            for validation_images, validation_targets in validation_data_loader:
                # input data as batch
                validation_images = list(image.to(device) for image in validation_images)
                validation_targets_device = [{k: v.to(device) for k, v in t.items()} for t in
                                             validation_targets]

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

                    # # Note that step should be called after validate()
                    # scheduler.step(loss_value)

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

                val_iter += 1

        except Exception as e:
            image_ids_error = extract_image_id_from_batch_using_dataset(validation_targets,
                                                                        validation_data_loader.dataset)
            raise ValidationError(image_ids_error, traceback.format_exc())

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
        validation_writer.add_scalar("rpn_loss_objectness", val_loss_objectness_agg,
                                     global_step=epoch)

        val_loss_rpn_box_reg_agg = running_validation_loss_rpn_box_reg / len(
            validation_data_loader.dataset)
        validation_loss_rpn_box_reg_arr.append(val_loss_rpn_box_reg_agg)
        validation_writer.add_scalar("rpn_loss_rpn_box_reg", val_loss_rpn_box_reg_agg,
                                     global_step=epoch)

        # split epoch into 10 equal parts and for each part and last epoch compute mAP and save
        # model
        if (epoch % save_step_size == 0) or (epoch == (EPOCHS - 1)):
            # set model to eval mode
            model.eval()

            # ----------------------------------mAP: TRAINING(unfiltered)---------------------------
            train_map = ZFTurbo_MAP_TRAINING(train_original_dimension, get_id_to_label_mAP())

            with torch.no_grad():
                # get data from train loader
                for images, targets in train_data_loader_mAP:
                    # send images to device
                    images = list(image.to(device) for image in images)

                    img_ids = extract_image_id_from_batch_using_dataset(targets,
                                                                        train_data_loader.dataset)

                    # add the target data
                    train_map.zfturbo_convert_targets_from_dataloader(targets, img_ids)

                    # get model outputs
                    outputs = model(images)

                    # add the output data
                    train_map.zfturbo_convert_outputs_from_model(outputs, img_ids)

            # compute the mAP
            mAP_train, ap_classes_train = train_map.zfturbo_compute_mAP()

            # print overall mAP and mAP for each class in console
            print(f"mAP for training at {epoch} is: {mAP_train}")
            print(f"mAP for training for all classes at {epoch} is: {ap_classes_train}")
            print()

            # save mAP as per counter level in tensorboard
            train_writer.add_scalar("mAP", mAP_train, global_step=model_save_counter)

            # --------------------------------mAP: VALIDATION(unfiltered)---------------------------
            validation_map = ZFTurbo_MAP_TRAINING(validation_original_dimension,
                                                  get_id_to_label_mAP())

            with torch.no_grad():
                # get data from validation loader
                for images, targets in validation_data_loader:
                    # send images to device
                    images = list(image.to(device) for image in images)

                    img_ids = extract_image_id_from_batch_using_dataset(targets,
                                                                        validation_data_loader.dataset)

                    # add the target data
                    validation_map.zfturbo_convert_targets_from_dataloader(targets, img_ids)

                    # get model outputs
                    outputs = model(images)

                    # add the output data
                    validation_map.zfturbo_convert_outputs_from_model(outputs, img_ids)

            # compute the mAP
            mAP_val, ap_classes_val = validation_map.zfturbo_compute_mAP()

            # print overall mAP and mAP for each class in console
            print(f"mAP for validation at {epoch} is: {mAP_val}")
            print(f"mAP for validation for all classes at {epoch} is: {ap_classes_val}")
            print()

            # save mAP as per counter level in tensorboard
            validation_writer.add_scalar("mAP", mAP_val, global_step=model_save_counter)

            # ------------------- SAVE MODEL --------------------------------------
            # last epoch
            if epoch == (EPOCHS - 1):
                # for last epoch save the state of model so that you can continue training the model
                saved_model_path = f"{saved_model_dir}/faster_rcnn_{model_save_counter}.pt"

                # https://debuggercafe.com/effective-model-saving-and-resuming-training-in-pytorch/
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler.state_dict(),
                }, saved_model_path)

            # other times
            else:
                # save only model state for each (epoch/10)th
                saved_model_path = f"{saved_model_dir}/faster_rcnn_{model_save_counter}.pt"
                torch.save(model.state_dict(), saved_model_path)

            # INCREMENT MODEL COUNTER
            model_save_counter += 1

    print("End time:" + str(datetime.now() - start))
    print("Program Complete")

    # tensorboard cleanup
    train_writer.flush()
    validation_writer.flush()

    train_writer.close()
    validation_writer.close()

except Exception as e:
    # print exception
    print(traceback.format_exc())

    # save model to resume training
    saved_model_path = f"{saved_model_dir}/faster_rcnn_error.pt"

    # https://debuggercafe.com/effective-model-saving-and-resuming-training-in-pytorch/
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'lr_scheduler': scheduler.state_dict(),
    }, saved_model_path)

# %% --------------------
print("-" * 25)
# print aggregates
# TRAIN
print(f"Train: Average: Sum Loss: {np.mean(sum_train_losses_arr)}")
print(f"Train: Average: Loss Classifier: {np.mean(train_loss_classifier_arr)}")
print(f"Train: Average: Loss Box Reg: {np.mean(train_loss_box_reg_arr)}")
print(f"Train: Average: RPN Loss Objectness: {np.mean(train_loss_objectness_arr)}")
print(f"Train: Average: Loss RPN Box Reg: {np.mean(train_loss_rpn_box_reg_arr)}")

# validation
print(f"Validation: Average: Sum Loss: {np.mean(sum_validation_losses_arr)}")
print(f"Validation: Average: Loss Classifier: {np.mean(validation_loss_classifier_arr)}")
print(f"Validation: Average: Loss Box Reg: {np.mean(validation_loss_box_reg_arr)}")
print(f"Validation: Average: RPN Loss Objectness: {np.mean(validation_loss_objectness_arr)}")
print(f"Validation: Average: Loss RPN Box Reg: {np.mean(validation_loss_rpn_box_reg_arr)}")
