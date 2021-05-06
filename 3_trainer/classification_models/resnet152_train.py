# %% --------------------
import os
import sys

# local
# BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------START HERE
# https://www.kaggle.com/corochann/vinbigdata-2-class-classifier-complete-pipeline
import albumentations
from common.classifier_models import initialize_model, get_param_to_optimize, \
    set_parameter_requires_grad
from common.utilities import UnNormalize
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from common.CustomDatasets import VBD_CXR_2_Class_Train
from collections import Counter
from torch.utils import tensorboard
from pathlib import Path
import shutil
from datetime import datetime

# %% --------------------set seeds
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

# %% --------------------DIRECTORIES and VARIABLES
IMAGE_DIR = f"{BASE_DIR}/input_data/512x512/train"
SPLIT_DIR = f"{BASE_DIR}/2_data_split"
SAVED_MODEL_DIRECTORY = f"{BASE_DIR}/4_saved_models/abnormality_detection_trained_models" \
                        f"/classification_models"

# %% --------------------TENSORBOARD DIRECTORY INITIALIZATION
train_tensorboard_dir = f"{BASE_DIR}/3_trainer/classification_models/resnet152/train"
os.makedirs(train_tensorboard_dir, exist_ok=True)

validation_tensorboard_dir = f"{BASE_DIR}/3_trainer/classification_models/resnet152/validation"
os.makedirs(validation_tensorboard_dir, exist_ok=True)

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
# https://albumentations.ai/docs/api_reference/augmentations/transforms/
train_transformer = albumentations.Compose([
    # augmentation operations
    albumentations.augmentations.transforms.RandomBrightnessContrast(p=0.3),
    albumentations.augmentations.transforms.ShiftScaleRotate(rotate_limit=5, p=0.4),
    # horizontal flipping
    albumentations.augmentations.transforms.HorizontalFlip(p=0.4),
    # resize operation
    albumentations.Resize(height=512, width=512, always_apply=True),
    # this normalization is performed based on ImageNet statistics per channel
    # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    albumentations.augmentations.transforms.Normalize()
])

validation_transformer = albumentations.Compose([
    # resize operation
    albumentations.Resize(height=512, width=512, always_apply=True),
    # this normalization is performed based on ImageNet statistics per channel
    # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    albumentations.augmentations.transforms.Normalize()
])

# %% --------------------DATASET
# 1 = abnormal
# 0 = normal
train_data_set = VBD_CXR_2_Class_Train(IMAGE_DIR,
                                       SPLIT_DIR + "/512/unmerged/90_percent_train"
                                                   "/2_class_classifier"
                                                   "/90_percent/train_df.csv", train_transformer)

validation_data_set = VBD_CXR_2_Class_Train(IMAGE_DIR,
                                            SPLIT_DIR + "/512/unmerged/90_percent_train"
                                                        "/2_class_classifier"
                                                        "/10_percent/holdout_df.csv",
                                            validation_transformer)

# %% --------------------WEIGHTED RANDOM SAMPLER
# weighted random sampler to handle class imbalance
# https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
# Get all the target classes
target_list = train_data_set.targets

# get the count
class_counts = Counter(target_list)

# Get the class weights. Class weights are the reciprocal of the number of items per class
class_weight = 1. / np.array(list(class_counts.values()))

# assign weights to each target
target_weight = []
for t in target_list:
    target_weight.append(class_weight[int(t)])

# create sampler based on weights
sampler = WeightedRandomSampler(weights=target_weight, num_samples=len(train_data_set),
                                replacement=True)

# %% --------------------DATALOADER
BATCH_SIZE = 8
workers = 4

# perform weighted random sampler for training only. NOTE: sampler shuffles the data by default
train_data_loader = torch.utils.data.DataLoader(
    train_data_set, batch_size=BATCH_SIZE, num_workers=workers, sampler=sampler)

validation_data_loader = torch.utils.data.DataLoader(
    validation_data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)

# %% --------------------WEIGHTS FOR LOSS
target_list = train_data_set.targets
class_counts = Counter(target_list)

# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# negative/positive, thus giving more weightage to positive class
pos_weight = class_counts[0] / class_counts[1]

# convert to Tensor vector where number of elements in vector = number of classes
# we have 1 class: 0 = Normal, 1 = Abnormal
pos_weight = torch.as_tensor(pos_weight, dtype=float)

# %% --------------------CRITERION
# https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
# https://visualstudiomagazine.com/Articles/2020/11/04/pytorch-training.aspx?Page=2
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# %% --------------------UnNormalize
unnormalizer = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# %% --------------------
# OVERFITTER
# train_image_ids, train_image, train_target = iter(train_data_loader).next()
# validation_image_ids, validation_image, validation_target = iter(validation_data_loader).next()

# %% --------------------
# define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# %% --------------------MODEL INSTANCE
# feature_extract_param = True means all layers frozen except the last user added layers
# feature_extract_param = False means all layers unfrozen and entire network learns new weights
# and biases
feature_extract_param = True

# 0 = normal CXR or 1 = abnormal CXR
# single label binary classifier
num_classes = 1

# input_size is minimum constraint
model, params_to_update = initialize_model("resnet152", num_classes, feature_extract_param,
                                           use_pretrained=True)

# %% --------------------HYPER-PARAMETERS
TOTAL_EPOCHS = 30
REDUCED_LR = 1e-4

# for first 5 EPOCHS train with all layers frozen except last, after that train with lowered LR
# with all layers unfrozen
INITIAL_EPOCHS = 5
INITIAL_LR = REDUCED_LR * 100

# %% --------------------OPTIMIZER
optimizer = torch.optim.Adam(params_to_update, lr=INITIAL_LR)

# %% --------------------move to device
model.to(device)

# %% --------------------TRAINING LOOP
print_iteration_frequency = 50
freeze_all_flag = True

train_iter = 0
val_iter = 0

# track train loss, validation loss, train accuracy, validation accuracy
val_acc_history_arr = []
train_acc_history_arr = []
train_loss_arr = []
valid_loss_arr = []

# condition to save model
lowest_loss = 10000000
best_model_found_epoch = 0

os.makedirs(SAVED_MODEL_DIRECTORY, exist_ok=True)
saved_model_path = f"{SAVED_MODEL_DIRECTORY}/resnet152.pt"

print("Program started")
# start time
start = datetime.now()
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
for epoch in range(TOTAL_EPOCHS):

    print('Epoch {}/{}'.format(epoch, TOTAL_EPOCHS - 1))
    print('-' * 10)

    if (epoch >= INITIAL_EPOCHS) and freeze_all_flag:
        # do below tasks only once when current epoch > INITIAL EPOCH
        freeze_all_flag = False

        # unfreeze all layers
        set_parameter_requires_grad(model, False)

        # get parameters to optimize
        params = get_param_to_optimize(model, False)

        # update optimizer and reduce LR
        optimizer = torch.optim.Adam(params, lr=REDUCED_LR)

    # ----------------------TRAINING PHASE----------------------
    model.train()

    # to track loss and accuracy for training phase at iteration level
    running_loss = 0.0
    running_corrects = 0

    # image grid flag
    train_epoch_flag = True

    # iterate the data
    # overfitting code
    # for _, images, targets in zip([train_image_ids], [train_image], [train_target]):
    for _, images, targets in train_data_loader:
        # send the input to device
        images = images.to(device)
        targets = targets.to(device)

        # visualize only the first batch in epoch to tensorboard
        if train_epoch_flag:
            # revert the normalization
            unnormalized_images = unnormalizer(images)

            # add images to tensorboard
            img_grid = torchvision.utils.make_grid(unnormalized_images)
            train_writer.add_image("train", img_tensor=img_grid, global_step=epoch)

        # turn off image_flag for other batches for current epoch
        train_epoch_flag = False

        # optimizer.zero_grad() is critically important because it resets all weight and bias
        # gradients to 0
        # we are updating W and B for each batch, thus zero the gradients in each batch
        optimizer.zero_grad()

        # forward pass
        # track the history in training mode
        with torch.set_grad_enabled(True):
            # make prediction
            outputs = model(images)

            # find loss
            train_loss = criterion(outputs.view(-1), targets)

            # converting logits to probabilities and keeping threshold of 0.5
            # https://discuss.pytorch.org/t/multilabel-classification-how-to-binarize-scores-how-to-learn-thresholds/25396
            preds = (torch.sigmoid(outputs.view(-1)) > 0.5).to(torch.float32)

            # loss.backward() method uses the back-propagation algorithm to compute all the
            # gradients associated with the weights and biases that a part of the network
            # containing loss_val
            train_loss.backward()

            # optimizer.step() statement uses the newly computed gradients to update all the
            # weights and biases in the neural network so that computed output values will get
            # closer to the target values
            optimizer.step()

        # iteration level statistics
        running_loss += train_loss.item() * images.size(0)
        running_corrects += torch.sum(preds == targets.data)

        if train_iter % print_iteration_frequency == 0:
            print(
                f'Train Iteration #{train_iter}:: {train_loss.item()} Acc: {torch.sum(preds == targets.data).item() / (len(preds))}')

        train_iter += 1

    # epoch level statistics take average of batch
    epoch_loss = running_loss / len(train_data_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_data_loader.dataset)
    print('Epoch Training:: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # track using tensorboard
    train_writer.add_scalar("loss", epoch_loss, global_step=epoch)
    train_writer.add_scalar("accuracy", epoch_acc, global_step=epoch)

    train_loss_arr.append(epoch_loss)
    train_acc_history_arr.append(epoch_acc.item())

    # ----------------------VALIDATION PHASE----------------------
    # https://visualstudiomagazine.com/Articles/2020/11/24/pytorch-accuracy.aspx?Page=2
    model.eval()

    # to track loss and accuracy for validation phase at iteration level
    running_loss = 0.0
    running_corrects = 0

    # iterate the data
    # overfitting code
    # for _, images, targets in zip([validation_image_ids], [validation_image], [validation_target]):
    for _, images, targets in validation_data_loader:
        # send the input to device
        images = images.to(device)
        targets = targets.to(device)

        # forward pass
        # dont track the history in validation mode
        with torch.set_grad_enabled(False):
            # make prediction
            outputs = model(images)

            # find loss
            val_loss = criterion(outputs.view(-1), targets)

            # converting logits to probabilities and keeping threshold of 0.5
            # https://discuss.pytorch.org/t/multilabel-classification-how-to-binarize-scores-how-to-learn-thresholds/25396
            preds = (torch.sigmoid(outputs.view(-1)) > 0.5).to(torch.float32)

        # iteration level statistics
        running_loss += val_loss.item() * images.size(0)
        running_corrects += torch.sum(preds == targets.data)

        if val_iter % print_iteration_frequency == 0:
            print(
                f'Validation Iteration #{val_iter}:: {val_loss.item()} Acc: {torch.sum(preds == targets.data).item() / (len(preds))}')

        val_iter += 1

    # epoch level statistics take average of batch
    epoch_loss = running_loss / len(validation_data_loader.dataset)
    epoch_acc = running_corrects.double() / len(validation_data_loader.dataset)
    print('Epoch Validation:: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # track using tensorboard
    validation_writer.add_scalar("loss", epoch_loss, global_step=epoch)
    validation_writer.add_scalar("accuracy", epoch_acc, global_step=epoch)

    valid_loss_arr.append(epoch_loss)
    val_acc_history_arr.append(epoch_acc.item())

    # save model based on lowest epoch validation loss
    if epoch_loss <= lowest_loss:
        lowest_loss = epoch_loss
        best_model_found_epoch = epoch
        # save model state based on best val accuracy per epoch
        # https://debuggercafe.com/effective-model-saving-and-resuming-training-in-pytorch/
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, saved_model_path)

print('Lowest Validation Acc: {:4f} at epoch:{}'.format(lowest_loss, best_model_found_epoch))

print("End time:" + str(datetime.now() - start))
print("Program Complete")

# tensorboard cleanup
train_writer.flush()
validation_writer.flush()

train_writer.close()
validation_writer.close()

# %% --------------------
# Average Loss
print("Average Train Loss:" + str(np.mean(train_loss_arr)))
print("Average Validation Loss:" + str(np.mean(valid_loss_arr)))

# Average Accuracy
print("Average Train Accuracy:" + str(np.mean(train_acc_history_arr)))
print("Average Validation Accuracy:" + str(np.mean(val_acc_history_arr)))
