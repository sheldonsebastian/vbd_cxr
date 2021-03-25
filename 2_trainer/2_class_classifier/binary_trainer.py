# %% --------------------
import os
import sys
from datetime import datetime

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
# https://www.kaggle.com/corochann/vinbigdata-2-class-classifier-complete-pipeline
import random
import albumentations
from common.classifier_models import initialize_model
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

# model name
# model_name = "alexnet"
# model_name = "vgg19"
model_name = "resnet152"

# %% --------------------TENSORBOARD DIRECTORY INITIALIZATION
train_tensorboard_dir = f"{TENSORBOARD_DIR}/2_class_classifier/{model_name}/train"
validation_tensorboard_dir = f"{TENSORBOARD_DIR}/2_class_classifier/{model_name}/validation"

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
    albumentations.augmentations.transforms.RandomBrightnessContrast(p=0.5),
    albumentations.augmentations.transforms.CoarseDropout(max_holes=8, max_height=25, max_width=25,
                                                          p=0.5),
    albumentations.augmentations.transforms.Blur(p=0.5, blur_limit=[3, 7]),
    albumentations.augmentations.transforms.RandomGamma(p=0.6, gamma_limit=[80, 120]),
    albumentations.augmentations.transforms.ShiftScaleRotate(scale_limit=0.15, rotate_limit=10,
                                                             p=0.5),
    albumentations.augmentations.transforms.Downscale(scale_min=0.25, scale_max=0.9, p=0.3),

    # horizontal flipping
    albumentations.augmentations.transforms.HorizontalFlip(p=0.5),

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
                                       MERGED_DIR + "/512/wbf_merged/90_percent_train"
                                                    "/2_class_classifier"
                                                    "/90_percent/train_df.csv",
                                       majority_transformations=train_transformer)

validation_data_set = VBD_CXR_2_Class_Train(IMAGE_DIR,
                                            MERGED_DIR + "/512/wbf_merged/90_percent_train"
                                                         "/2_class_classifier"
                                                         "/10_percent/holdout_df.csv",
                                            majority_transformations=validation_transformer)

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
BATCH_SIZE = 32
workers = int(os.getenv("NUM_WORKERS"))

# # perform weighted random sampler for training only. NOTE: sampler shuffles the data by default
train_data_loader = torch.utils.data.DataLoader(
    train_data_set, batch_size=BATCH_SIZE, num_workers=workers, sampler=sampler)

validation_data_loader = torch.utils.data.DataLoader(
    validation_data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

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
model = initialize_model(model_name, num_classes, feature_extract_param,
                         use_pretrained=True)

# %% --------------------HYPER-PARAMETERS
LR = 1e-3
EPOCHS = 50

# %% --------------------OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

# %% --------------------move to device
model.to(device)

# %% --------------------CHECK THE DATALOADER
# # works when num of workers = 0 on local environment
# for images, targets in train_data_loader:
#     print(images)
#     print(targets)

# %% --------------------UnNormalize
unnormalizer = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# %% --------------------TRAINING LOOP
print("Program started")

train_iter = 0
val_iter = 0

# start time
start = datetime.now()

# track train loss, validation loss, train accuracy, validation accuracy
val_acc_history_arr = []
train_acc_history_arr = []
train_loss_arr = []
valid_loss_arr = []

best_acc = 0.0

# if directory does not exist then create it
saved_model_dir = Path(f"{SAVED_MODEL_DIR}/2_class_classifier/{model_name}")

if not saved_model_dir.exists():
    os.makedirs(saved_model_dir)

saved_model_path = f"{saved_model_dir}/{model_name}.pt"

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
for epoch in range(EPOCHS):

    print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
    print('-' * 10)

    # ----------------------TRAINING PHASE----------------------
    model.train()

    # to track loss and accuracy for training phase at iteration level
    running_loss = 0.0
    running_corrects = 0

    # image grid flag
    train_epoch_flag = True

    # iterate the data
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

        if train_iter % 50 == 0:
            print(
                f'Train Iteration #{train_iter}:: {train_loss.item()} Acc: {torch.sum(preds == targets.data).item() / (len(preds))}')

        train_iter += 1

    # epoch level statistics take average of batch
    epoch_loss = running_loss / len(train_data_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_data_loader.dataset)
    print('Training:: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

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

        if val_iter % 50 == 0:
            print(
                f'Validation Iteration #{val_iter}:: {val_loss.item()} Acc: {torch.sum(preds == targets.data).item() / (len(preds))}')

        val_iter += 1

    # epoch level statistics take average of batch
    epoch_loss = running_loss / len(validation_data_loader.dataset)
    epoch_acc = running_corrects.double() / len(validation_data_loader.dataset)
    print('Validation:: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # track using tensorboard
    validation_writer.add_scalar("loss", epoch_loss, global_step=epoch)
    validation_writer.add_scalar("accuracy", epoch_acc, global_step=epoch)

    valid_loss_arr.append(epoch_loss)
    val_acc_history_arr.append(epoch_acc.item())

    # model checkpoint, save best model only when epoch accuracy in validation is best
    if epoch_acc > best_acc:
        best_acc = epoch_acc

        # save model state based on best val accuracy per epoch
        # https://debuggercafe.com/effective-model-saving-and-resuming-training-in-pytorch/
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, saved_model_path)

print('Best Validation Acc: {:4f}'.format(best_acc))

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
