# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr"
# cerberus
# BASE_DIR = "/home/ssebastian94/vbd_cxr"

# add HOME DIR to PYTHONPATH
sys.path.append(BASE_DIR)

# %% --------------------
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, \
    precision_score, recall_score, precision_recall_curve, auc
import pandas as pd

pd.set_option('display.max_columns', None)

# %% --------------------
SPLIT_DIR = f"{BASE_DIR}/2_data_split"
output_dir = f"{BASE_DIR}/5_inference_on_holdout_10_percent/0_predictions"

# %% --------------------
# read resnet152 output
resnet_152 = pd.read_csv(f"{output_dir}/holdout_resnet152.csv")
resnet_152.columns = ['image_id', 'target_resnet_152', 'probabilities_resnet_152']

# read vgg19 output
vgg_19 = pd.read_csv(f"{output_dir}/holdout_vgg19.csv")
vgg_19.columns = ['image_id', 'target_vgg_19', 'probabilities_vgg_19']

# %% --------------------
# join dataframes using image_id as key
combined = resnet_152.join(vgg_19.set_index("image_id"), on="image_id")

combined = combined[
    ["image_id", "probabilities_resnet_152", "probabilities_vgg_19"]]

# taking average of all models
combined["probabilities"] = (combined["probabilities_resnet_152"] + combined[
    "probabilities_vgg_19"]) / 2

# %% --------------------
combined = combined[["image_id", "probabilities"]]

# %% --------------------
# save output of classification ensemble
combined.to_csv(output_dir + "/holdout_ensemble_classification.csv", index=False)

# %% --------------------EVALUATION
print("Holdout 10% Evaluation for Ensemble of Resnet152 and VGG19")

# %% --------------------
combined["ground_truth"] = -1

# read gt
holdout_gt = pd.read_csv(f"{SPLIT_DIR}/512/unmerged/10_percent_holdout/holdout_df.csv")

# %% --------------------
# add GT to prediction dataframe
for image_id in combined["image_id"].unique():
    # in ground truth we are finding no findings class
    if (holdout_gt[holdout_gt["image_id"] == image_id]["class_id"] == 14).any():
        combined.loc[combined["image_id"] == image_id, "ground_truth"] = 0
    else:
        combined.loc[combined["image_id"] == image_id, "ground_truth"] = 1

# %% --------------------confidence threshold
aggregated_results = []
for conf_thr in [0.10, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    predictions = [1 if p >= conf_thr else 0 for p in combined["probabilities"]]

    # %% --------------------Accuracy
    accuracy_score_holdout = accuracy_score(combined["ground_truth"], predictions)

    # --------------------Precision Score
    precision_holdout = precision_score(combined["ground_truth"], predictions)

    # --------------------Recall Score
    recall_holdout = recall_score(combined["ground_truth"], predictions)

    # --------------------F1 Score
    f1_holdout = f1_score(combined["ground_truth"], predictions)

    # --------------------PR AUC Curve
    # ROC is computed using fpr and tpr
    precision, recall, thresholds = precision_recall_curve(combined["ground_truth"],
                                                           predictions)
    precision_recall_score = auc(recall, precision)

    # --------------------ROC AUC Curve
    # ROC is computed using fpr and tpr
    roc_score_holdout = roc_auc_score(combined["ground_truth"], predictions)

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
