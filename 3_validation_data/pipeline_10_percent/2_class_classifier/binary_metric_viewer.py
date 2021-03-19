# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
# env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# %% --------------------
# DIRECTORIES
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
MERGED_DIR = os.getenv("MERGED_DIR")

# %% --------------------START HERE
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, accuracy_score, \
    precision_score, recall_score
import pandas as pd
from common.utilities import confusion_matrix_plotter, plot_roc_cur

# %% --------------------
print("Holdout 10% data")
# read the predicted csv files for 10% holdout set
holdout_pred = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/pipeline_10_percent/2_class_classifier/predictions/archives"
                                "/resnet_50_augmentations/holdout_resnet50_vanilla.csv")
holdout_pred["ground_truth"] = -1

holdout_gt = pd.read_csv(MERGED_DIR + f"/wbf_merged/holdout_df.csv")

# %% --------------------
for image_id in holdout_pred["image_id"].unique():
    # in ground truth we are finding no findings class
    if (holdout_gt[holdout_gt["image_id"] == image_id]["class_id"] == 14).any():
        holdout_pred.loc[holdout_pred["image_id"] == image_id, "ground_truth"] = 0
    else:
        holdout_pred.loc[holdout_pred["image_id"] == image_id, "ground_truth"] = 1

# %% --------------------Accuracy
accuracy_score_holdout = accuracy_score(holdout_pred["ground_truth"], holdout_pred["target"])
print(f"Accuracy score holdout :" + str(accuracy_score_holdout))

# --------------------confusion matrix
confusion_matrix_holdout = confusion_matrix(holdout_pred["ground_truth"],
                                            holdout_pred["target"])
confusion_matrix_plotter(confusion_matrix_holdout, f"Confusion Matrix Holdout")

# --------------------Precision Score
precision_holdout = precision_score(holdout_pred["ground_truth"],
                                    holdout_pred["target"])
print(f"Precision score holdout :" + str(precision_holdout))

# --------------------Recall Score
recall_holdout = recall_score(holdout_pred["ground_truth"],
                              holdout_pred["target"])
print(f"Recall score holdout :" + str(recall_holdout))

# --------------------F1 Score
f1_holdout = f1_score(holdout_pred["ground_truth"],
                      holdout_pred["target"])
print(f"F1 score holdout :" + str(f1_holdout))

# --------------------ROC AUC Curve
roc_score_holdout = roc_auc_score(holdout_pred["ground_truth"],
                                  holdout_pred["target"])
roc_auc_holdout = roc_curve(holdout_pred["ground_truth"],
                            holdout_pred["target"])
print(f"ROC score holdout :" + str(roc_score_holdout))
plot_roc_cur(roc_auc_holdout[0], roc_auc_holdout[1], f"ROC Holdout :{roc_score_holdout}")
