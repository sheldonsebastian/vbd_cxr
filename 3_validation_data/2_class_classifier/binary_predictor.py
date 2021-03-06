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
# perform for validation and holdout sets for respected fold
import random
import albumentations
import numpy as np
import torch
from common.CustomDatasets import VBD_CXR_2_Class_Train
from common.classifier_models import initialize_model
from datetime import datetime
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
MERGED_DIR = os.getenv("MERGED_DIR")
SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR")
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")

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

# %% --------------------
# generic transformer used for validation data and holdout data
generic_transformer = albumentations.Compose([
    # resize operation
    albumentations.Resize(height=512, width=512, always_apply=True),

    # this normalization is performed based on ImageNet statistics per channel
    albumentations.augmentations.transforms.Normalize()
])

# %% --------------------DATASET
# create validation dataset
validation_data_set = VBD_CXR_2_Class_Train(IMAGE_DIR,
                                            MERGED_DIR + "/wbf_merged/k_fold_splits"
                                                         "/2_class_classifier"
                                                         "/validation_df_5_folds.csv",
                                            majority_transformations=generic_transformer,
                                            fold=fold)

# create holdout dataset
# holdout does not have fold, thus use all data
holdout_data_set = VBD_CXR_2_Class_Train(IMAGE_DIR,
                                         MERGED_DIR + "/wbf_merged/k_fold_splits"
                                                      "/2_class_classifier"
                                                      "/holdout.csv",
                                         majority_transformations=generic_transformer,
                                         fold=None)

# %% --------------------DATALOADER
BATCH_SIZE = 32
workers = int(os.getenv("NUM_WORKERS"))

# create dataloader
validation_data_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=BATCH_SIZE,
                                                     shuffle=False, num_workers=workers)

holdout_data_loader = torch.utils.data.DataLoader(holdout_data_set, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=workers)

# %% --------------------
# define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# %% --------------------MODEL INSTANCE
# create model instance
# model name
model_name = "resnet18"

# feature_extract_param = True means all layers frozen except the last user added layers
# feature_extract_param = False means all layers unfrozen and entire network learns new weights
# and biases
feature_extract_param = True

# 0 = normal CXR or 1 = abnormal CXR
# single label binary classifier
num_classes = 1

# initializing model
model, input_size = initialize_model(model_name, num_classes, feature_extract_param,
                                     use_pretrained=True)

# load model weights
saved_model_path = f"{SAVED_MODEL_DIR}/2_class_classifier/{fold}/{model_name}.pt"
model.load_state_dict(torch.load(saved_model_path, map_location=torch.device(device)))

# %% --------------------
# set model to eval mode, to not disturb the weights
model.eval()

# %% --------------------
# send model to device
model = model.to(device)

# %% --------------------
# make predictions for validation data
print("Validation predictions started")
# start time
start = datetime.now()

# arrays
image_id_arr = []
pred_label_arr = []
# ground truth
target_label_arr = []

val_iter = 0

with torch.no_grad():
    for image_ids, images, targets in validation_data_loader:
        # send the input to device
        images = images.to(device)
        targets = targets.to(device)

        # forward pass
        # dont track the history in validation mode
        with torch.set_grad_enabled(False):
            # make prediction
            outputs = model(images)

            # converting logits to probabilities and keeping threshold of 0.5
            # https://discuss.pytorch.org/t/multilabel-classification-how-to-binarize-scores-how-to-learn-thresholds/25396
            preds = (torch.sigmoid(outputs.view(-1)) > 0.5).to(torch.float32)

        # TODO iterate preds, image_ids, and targets and add them to the csv file

        if val_iter % 50 == 0:
            print(f"Iteration #: {val_iter}")

        val_iter += 1

print("Predictions Complete")
print("End time:" + str(datetime.now() - start))

val_predictions = pd.DataFrame({"image_id": image_id_arr,
                                "target": pred_label_arr,
                                "prediction": target_label_arr})

# %% --------------------
# validation path
validation_path = f"{VALIDATION_PREDICTION_DIR}/2_class_classifier/predictions/validation_predictions_{fold}.csv"

# write csv file
val_predictions.to_csv(validation_path, index=False)

# %% --------------------
# make predictions for holdout data
print("Holdout predictions started")
# start time
start = datetime.now()

# arrays
image_id_arr = []
pred_label_arr = []
# ground truth
target_label_arr = []

holdout_iter = 0

with torch.no_grad():
    for image_ids, images, targets in holdout_data_loader:
        # send the input to device
        images = images.to(device)
        targets = targets.to(device)

        # forward pass
        # dont track the history in validation mode
        with torch.set_grad_enabled(False):
            # make prediction
            outputs = model(images)

            # converting logits to probabilities and keeping threshold of 0.5
            # https://discuss.pytorch.org/t/multilabel-classification-how-to-binarize-scores-how-to-learn-thresholds/25396
            preds = (torch.sigmoid(outputs.view(-1)) > 0.5).to(torch.float32)

        # TODO iterate preds, image_ids, and targets and add them to the csv file

        if holdout_iter % 50 == 0:
            print(f"Iteration #: {holdout_iter}")

        holdout_iter += 1

print("Predictions Complete")
print("End time:" + str(datetime.now() - start))

holdout_predictions = pd.DataFrame({"image_id": image_id_arr,
                                    "target": pred_label_arr,
                                    "prediction": target_label_arr})

# %% --------------------
# holdout path
holdout_path = f"{VALIDATION_PREDICTION_DIR}/2_class_classifier/predictions/holdout_prediction_{fold}.csv"

# write csv file
holdout_predictions.to_csv(holdout_path, index=False)
