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

# DIRECTORIES
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")

# %% --------------------START HERE
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import pandas as pd
from common.utilities import confusion_matrix_plotter, plot_roc_cur

# %% --------------------
# read the predicted csv files for validation and holdout set
validation_fold_0 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/validation_predictions_0.csv")
holdout_fold_0 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/holdout_0.csv")

validation_fold_1 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/validation_predictions_1.csv")
holdout_fold_1 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/holdout_1.csv")

validation_fold_2 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/validation_predictions_2.csv")
holdout_fold_2 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/holdout_2.csv")

validation_fold_3 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/validation_predictions_3.csv")
holdout_fold_3 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/holdout_3.csv")

validation_fold_4 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/validation_predictions_4.csv")
holdout_fold_4 = pd.read_csv(
    VALIDATION_PREDICTION_DIR + "/2_class_classifier/predictions/holdout_4.csv")

# %% --------------------VALIDATION
# confusion matrix
confusion_matrix_val_0 = confusion_matrix(validation_fold_0["target"],
                                          validation_fold_0["prediction"])
confusion_matrix_plotter(confusion_matrix_val_0, "Confusion Matrix Validation Fold 0")

confusion_matrix_val_1 = confusion_matrix(validation_fold_1["target"],
                                          validation_fold_1["prediction"])
confusion_matrix_plotter(confusion_matrix_val_1, "Confusion Matrix Validation Fold 1")

confusion_matrix_val_2 = confusion_matrix(validation_fold_2["target"],
                                          validation_fold_2["prediction"])
confusion_matrix_plotter(confusion_matrix_val_2, "Confusion Matrix Validation Fold 2")

confusion_matrix_val_3 = confusion_matrix(validation_fold_3["target"],
                                          validation_fold_3["prediction"])
confusion_matrix_plotter(confusion_matrix_val_3, "Confusion Matrix Validation Fold 3")

confusion_matrix_val_4 = confusion_matrix(validation_fold_4["target"],
                                          validation_fold_4["prediction"])
confusion_matrix_plotter(confusion_matrix_val_4, "Confusion Matrix Validation Fold 4")

# %% --------------------HOLDOUT
# confusion matrix
confusion_matrix_holdout_0 = confusion_matrix(holdout_fold_0["target"],
                                              holdout_fold_0["prediction"])
confusion_matrix_plotter(confusion_matrix_holdout_0, "Confusion Matrix Holdout Fold 0")

confusion_matrix_holdout_1 = confusion_matrix(holdout_fold_1["target"],
                                              holdout_fold_1["prediction"])
confusion_matrix_plotter(confusion_matrix_holdout_1, "Confusion Matrix Holdout Fold 1")

confusion_matrix_holdout_2 = confusion_matrix(holdout_fold_2["target"],
                                              holdout_fold_2["prediction"])
confusion_matrix_plotter(confusion_matrix_holdout_2, "Confusion Matrix Holdout Fold 2")

confusion_matrix_holdout_3 = confusion_matrix(holdout_fold_3["target"],
                                              holdout_fold_3["prediction"])
confusion_matrix_plotter(confusion_matrix_holdout_3, "Confusion Matrix Holdout Fold 3")

confusion_matrix_holdout_4 = confusion_matrix(holdout_fold_4["target"],
                                              holdout_fold_4["prediction"])
confusion_matrix_plotter(confusion_matrix_holdout_4, "Confusion Matrix Holdout Fold 4")

# %% --------------------VALIDATION
# F1 Score
f1_val_0 = f1_score(validation_fold_0["target"],
                    validation_fold_0["prediction"])
print("F1 score validation fold 0:" + str(f1_val_0))

f1_val_1 = f1_score(validation_fold_1["target"],
                    validation_fold_1["prediction"])
print("F1 score validation fold 1:" + str(f1_val_1))

f1_val_2 = f1_score(validation_fold_2["target"],
                    validation_fold_2["prediction"])
print("F1 score validation fold 2:" + str(f1_val_2))

f1_val_3 = f1_score(validation_fold_3["target"],
                    validation_fold_3["prediction"])
print("F1 score validation fold 3:" + str(f1_val_3))

f1_val_4 = f1_score(validation_fold_4["target"],
                    validation_fold_4["prediction"])
print("F1 score validation fold 4:" + str(f1_val_4))

# %% --------------------HOLDOUT
# F1 Score
f1_holdout_0 = f1_score(holdout_fold_0["target"],
                        holdout_fold_0["prediction"])
print("F1 score holdout fold 0:" + str(f1_holdout_0))

f1_holdout_1 = f1_score(holdout_fold_1["target"],
                        holdout_fold_1["prediction"])
print("F1 score holdout fold 1:" + str(f1_holdout_1))

f1_holdout_2 = f1_score(holdout_fold_2["target"],
                        holdout_fold_2["prediction"])
print("F1 score holdout fold 2:" + str(f1_holdout_2))

f1_holdout_3 = f1_score(holdout_fold_3["target"],
                        holdout_fold_3["prediction"])
print("F1 score holdout fold 3:" + str(f1_holdout_3))

f1_holdout_4 = f1_score(holdout_fold_4["target"],
                        holdout_fold_4["prediction"])
print("F1 score holdout fold 4:" + str(f1_holdout_4))

# %% --------------------VALIDATION
# ROC AUC Curve
roc_score_val_0 = roc_auc_score(validation_fold_0["target"],
                                validation_fold_0["prediction"])
roc_auc_val_0 = roc_curve(validation_fold_0["target"],
                          validation_fold_0["prediction"])
print("ROC score validation fold 0:" + str(roc_score_val_0))
plot_roc_cur(roc_auc_val_0[0], roc_auc_val_0[1], f"ROC Validation Fold 0:{roc_score_val_0}")

roc_score_val_1 = roc_auc_score(validation_fold_1["target"],
                                validation_fold_1["prediction"])
roc_auc_val_1 = roc_curve(validation_fold_1["target"],
                          validation_fold_1["prediction"])
print("ROC score validation fold 1:" + str(roc_score_val_1))
plot_roc_cur(roc_auc_val_1[0], roc_auc_val_1[1], f"ROC Validation Fold 1:{roc_score_val_1}")

roc_score_val_2 = roc_auc_score(validation_fold_2["target"],
                                validation_fold_2["prediction"])
roc_auc_val_2 = roc_curve(validation_fold_2["target"],
                          validation_fold_2["prediction"])
print("ROC score validation fold 2:" + str(roc_score_val_2))
plot_roc_cur(roc_auc_val_2[0], roc_auc_val_2[1], f"ROC Validation Fold 2:{roc_score_val_2}")

roc_score_val_3 = roc_auc_score(validation_fold_3["target"],
                                validation_fold_3["prediction"])
roc_auc_val_3 = roc_curve(validation_fold_3["target"],
                          validation_fold_3["prediction"])
print("ROC score validation fold 3:" + str(roc_score_val_3))
plot_roc_cur(roc_auc_val_3[0], roc_auc_val_3[1], f"ROC Validation Fold 3:{roc_score_val_3}")

roc_score_val_4 = roc_auc_score(validation_fold_4["target"],
                                validation_fold_4["prediction"])
roc_auc_val_4 = roc_curve(validation_fold_4["target"],
                          validation_fold_4["prediction"])
print("ROC score validation fold 4:" + str(roc_score_val_4))
plot_roc_cur(roc_auc_val_4[0], roc_auc_val_4[1], f"ROC Validation Fold 4:{roc_score_val_4}")

# %% --------------------HOLDOUT
# ROC AUC Curve
roc_score_holdout_0 = roc_auc_score(holdout_fold_0["target"],
                                    holdout_fold_0["prediction"])
roc_auc_holdout_0 = roc_curve(holdout_fold_0["target"],
                              holdout_fold_0["prediction"])
print("ROC score holdout fold 0:" + str(roc_score_holdout_0))
plot_roc_cur(roc_auc_holdout_0[0], roc_auc_holdout_0[1],
             f"ROC Holdout Fold 0:{roc_score_holdout_0}")

roc_score_holdout_1 = roc_auc_score(holdout_fold_1["target"],
                                    holdout_fold_1["prediction"])
roc_auc_holdout_1 = roc_curve(holdout_fold_1["target"],
                              holdout_fold_1["prediction"])
print("ROC score holdout fold 1:" + str(roc_score_holdout_1))
plot_roc_cur(roc_auc_holdout_1[0], roc_auc_holdout_1[1],
             f"ROC Holdout Fold 1:{roc_score_holdout_1}")

roc_score_holdout_2 = roc_auc_score(holdout_fold_2["target"],
                                    holdout_fold_2["prediction"])
roc_auc_holdout_2 = roc_curve(holdout_fold_2["target"],
                              holdout_fold_2["prediction"])
print("ROC score holdout fold 2:" + str(roc_score_holdout_2))
plot_roc_cur(roc_auc_holdout_2[0], roc_auc_holdout_2[1],
             f"ROC Holdout Fold 2:{roc_score_holdout_2}")

roc_score_holdout_3 = roc_auc_score(holdout_fold_3["target"],
                                    holdout_fold_3["prediction"])
roc_auc_holdout_3 = roc_curve(holdout_fold_3["target"],
                              holdout_fold_3["prediction"])
print("ROC score holdout fold 3:" + str(roc_score_holdout_3))
plot_roc_cur(roc_auc_holdout_3[0], roc_auc_holdout_3[1],
             f"ROC Holdout Fold 3:{roc_score_holdout_3}")

roc_score_holdout_4 = roc_auc_score(holdout_fold_4["target"],
                                    holdout_fold_4["prediction"])
roc_auc_holdout_4 = roc_curve(holdout_fold_4["target"],
                              holdout_fold_4["prediction"])
print("ROC score holdout fold 4:" + str(roc_score_holdout_4))
plot_roc_cur(roc_auc_holdout_4[0], roc_auc_holdout_4[1],
             f"ROC Holdout Fold 4:{roc_score_holdout_4}")
