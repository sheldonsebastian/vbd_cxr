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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, \
    precision_score, recall_score, precision_recall_curve, auc
import pandas as pd

pd.set_option('display.max_columns', None)

print("Holdout 10% data")

model_name = "vgg19"

# %% --------------------
# read predictions
holdout_pred = pd.read_csv(
    VALIDATION_PREDICTION_DIR + f"/pipeline_10_percent/2_class_classifier/predictions/holdout_{model_name}.csv")
holdout_pred = pd.read_csv("D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/final_outputs/holdout/holdout_binary_ensembled.csv")

holdout_pred["ground_truth"] = -1

# read gt
holdout_gt = pd.read_csv(MERGED_DIR + f"/512/unmerged/10_percent_holdout/holdout_df.csv")

# %% --------------------
# add GT to prediction dataframe
for image_id in holdout_pred["image_id"].unique():
    # in ground truth we are finding no findings class
    if (holdout_gt[holdout_gt["image_id"] == image_id]["class_id"] == 14).any():
        holdout_pred.loc[holdout_pred["image_id"] == image_id, "ground_truth"] = 0
    else:
        holdout_pred.loc[holdout_pred["image_id"] == image_id, "ground_truth"] = 1

# %% --------------------confidence threshold
aggregated_results = []
for conf_thr in [0.10, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    predictions = [1 if p >= conf_thr else 0 for p in holdout_pred["probabilities"]]

    # %% --------------------Accuracy
    accuracy_score_holdout = accuracy_score(holdout_pred["ground_truth"], predictions)

    # --------------------Precision Score
    precision_holdout = precision_score(holdout_pred["ground_truth"], predictions)

    # --------------------Recall Score
    recall_holdout = recall_score(holdout_pred["ground_truth"], predictions)

    # --------------------F1 Score
    f1_holdout = f1_score(holdout_pred["ground_truth"], predictions)

    # --------------------PR AUC Curve
    # ROC is computed using fpr and tpr
    precision, recall, thresholds = precision_recall_curve(holdout_pred["ground_truth"],
                                                           predictions)
    precision_recall_score = auc(recall, precision)

    # --------------------ROC AUC Curve
    # ROC is computed using fpr and tpr
    roc_score_holdout = roc_auc_score(holdout_pred["ground_truth"], predictions)

    aggregated_results.append({"Confidence": conf_thr,
                               "Accuracy": accuracy_score_holdout,
                               "Precision": precision_holdout,
                               "Recall": recall_holdout,
                               "F1": f1_holdout,
                               "ROC AUC": roc_score_holdout,
                               "PR AUC": precision_recall_score
                               })
# %% --------------------
agg_df = pd.DataFrame(aggregated_results,
                      columns=["Confidence", "Accuracy", "Precision", "Recall",
                               "F1", "ROC AUC", "PR AUC"])

# %% --------------------
print(agg_df.sort_values(["F1"], ascending=False))
