# %% --------------------
import sys

# local
# BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------START HERE
import albumentations
import torch
from common.CustomDatasets import VBD_CXR_2_Class_Test
from common.classifier_models import initialize_model
from datetime import datetime
import pandas as pd
from pathlib import Path
import os

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

# %% --------------------
# generic transformer used for validation data and holdout data
generic_transformer = albumentations.Compose([
    # this normalization is performed based on ImageNet statistics per channel
    albumentations.augmentations.transforms.Normalize()
])

# %% --------------------DATASET
# create holdout dataset
# holdout was 10% split
holdout_data_set = VBD_CXR_2_Class_Test(IMAGE_DIR,
                                        SPLIT_DIR + "/512/unmerged/10_percent_holdout/holdout_df.csv",
                                        generic_transformer)

# %% --------------------DATALOADER
BATCH_SIZE = 32
workers = 4

# create dataloader
holdout_data_loader = torch.utils.data.DataLoader(holdout_data_set, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=workers)

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

# initializing model
model, params_to_update = initialize_model("resnet152", num_classes, feature_extract_param,
                                           use_pretrained=True)

# load model weights
saved_model_path = f"{SAVED_MODEL_DIRECTORY}/resnet152.pt"
model.load_state_dict(
    torch.load(saved_model_path, map_location=torch.device(device))["model_state_dict"])

# %% --------------------
# set model to eval mode, to not disturb the weights
model.eval()

# %% --------------------
# send model to device
model = model.to(device)

# %% --------------------
# make predictions for holdout data
print("Holdout predictions started")
# start time
start = datetime.now()

# arrays
image_id_arr = []
pred_label_arr = []
# save probabilities and targets
pred_prob_arr = []
holdout_iter = 0

with torch.no_grad():
    for image_ids, images in holdout_data_loader:
        # send the input to device
        images = images.to(device)

        # forward pass
        # dont track the history in validation mode
        with torch.set_grad_enabled(False):
            # make prediction
            outputs = model(images)
            probabilities = torch.sigmoid(outputs.view(-1))

            # converting logits to probabilities and keeping threshold of 0.5
            # https://discuss.pytorch.org/t/multilabel-classification-how-to-binarize-scores-how-to-learn-thresholds/25396
            preds = (probabilities > 0.5).to(torch.float32)

        # iterate preds, image_ids and add them to the csv file
        for img_id, p, prob in zip(image_ids, preds, probabilities):
            image_id_arr.append(img_id)
            pred_label_arr.append(p.item())
            pred_prob_arr.append(prob.item())

        if holdout_iter % 50 == 0:
            print(f"Iteration #: {holdout_iter}")

        holdout_iter += 1

print("Predictions Complete")
print("End time:" + str(datetime.now() - start))

holdout_predictions = pd.DataFrame({"image_id": image_id_arr,
                                    "target": pred_label_arr,
                                    "probabilities": pred_prob_arr})

# %% --------------------
# holdout path
holdout_path = f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions"

if not Path(holdout_path).exists():
    os.makedirs(holdout_path)

# write csv file
holdout_predictions.to_csv(holdout_path + "/holdout_resnet152.csv", index=False)
