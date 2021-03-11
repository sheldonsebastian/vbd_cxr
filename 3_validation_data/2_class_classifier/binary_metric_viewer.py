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
validation_fold = pd.read_csv(
    VALIDATION_PREDICTION_DIR + f"/2_class_classifier/predictions/validation_predictions.csv")
holdout_fold = pd.read_csv(
    VALIDATION_PREDICTION_DIR + f"/2_class_classifier/predictions/holdout.csv")

# --------------------VALIDATION
# confusion matrix
confusion_matrix_val = confusion_matrix(validation_fold["target"],
                                        validation_fold["prediction"])
confusion_matrix_plotter(confusion_matrix_val, f"Confusion Matrix Validation Fold")

# --------------------HOLDOUT
# confusion matrix
confusion_matrix_holdout = confusion_matrix(holdout_fold["target"],
                                            holdout_fold["prediction"])
confusion_matrix_plotter(confusion_matrix_holdout, f"Confusion Matrix Holdout Fold")

# --------------------VALIDATION
# F1 Score
f1_val = f1_score(validation_fold["target"],
                  validation_fold["prediction"])
print(f"F1 score validation fold :" + str(f1_val))

# --------------------HOLDOUT
# F1 Score
f1_holdout = f1_score(holdout_fold["target"],
                      holdout_fold["prediction"])
print(f"F1 score holdout fold :" + str(f1_holdout))

# --------------------VALIDATION
# ROC AUC Curve
roc_score_val = roc_auc_score(validation_fold["target"],
                              validation_fold["prediction"])
roc_auc_val = roc_curve(validation_fold["target"],
                        validation_fold["prediction"])
print(f"ROC score validation fold :" + str(roc_score_val))
plot_roc_cur(roc_auc_val[0], roc_auc_val[1], f"ROC Validation Fold :{roc_score_val}")

# --------------------HOLDOUT
# ROC AUC Curve
roc_score_holdout = roc_auc_score(holdout_fold["target"],
                                  holdout_fold["prediction"])
roc_auc_holdout = roc_curve(holdout_fold["target"],
                            holdout_fold["prediction"])
print(f"ROC score holdout fold :" + str(roc_score_holdout))
plot_roc_cur(roc_auc_holdout[0], roc_auc_holdout[1], f"ROC Holdout Fold :{roc_score_holdout}")