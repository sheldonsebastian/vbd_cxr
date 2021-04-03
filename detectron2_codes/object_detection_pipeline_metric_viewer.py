# # %% --------------------
# import os
# import sys
#
# from dotenv import load_dotenv
#
# # local
# env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
#            "Workspace/vbd_cxr/6_environment_files/local.env "
# # cerberus
# # env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"
#
# load_dotenv(env_file)
#
# # add HOME DIR to PYTHONPATH
# sys.path.append(os.getenv("HOME_DIR"))
#
# # %% --------------------imports
# import pandas as pd
# from common.detectron2_post_processor_utils import post_process_conf_filter_nms, \
#     post_process_conf_filter_wbf
# from common.mAP_utils import normalize_bb, zfturbo_compute_mAP
#
# # %% --------------------directories
# BINARY_PRED_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
# DETECTRON2_DIR = os.getenv("DETECTRON2_DIR")
# MERGED_DIR = os.getenv("MERGED_DIR")
#
# # %% --------------------
# binary_conf_thr = 0.6
# obj_det_conf_thr = 0.10
# iou_threshold = 0.4
#
# # %% --------------------read the predictions
# # 2 class filter prediction
# binary_prediction = pd.read_csv(
#     BINARY_PRED_DIR + "/pipeline_10_percent/2_class_classifier/predictions/holdout_resnet152.csv")
#
# # object detection predictions
# object_detection_prediction = pd.read_csv(
#     DETECTRON2_DIR + "/faster_rcnn/holdout/unmerged_external/holdout.csv")
#
# # %% --------------------
# # GROUND TRUTH
# original_dataset = pd.read_csv(MERGED_DIR + "/512/wbf_merged/10_percent_holdout/holdout_df_0_6.csv")
#
# # %% --------------------
# # get all image ids in original dataset
# original_image_ids = list(original_dataset["image_id"].unique())
#
# # %% --------------------
# # UPDATE TARGET OF BINARY CLASSIFIER USING CONF THR
# updated_targets = [1 if p >= binary_conf_thr else 0 for p in binary_prediction["probabilities"]]
# binary_prediction["target"] = updated_targets
#
# # %% --------------------
# abnormal_ids = list(binary_prediction[binary_prediction["target"] == 1]["image_id"].unique())
#
# # %% --------------------
# # subset the object detections based on binary classifier
# object_detection_prediction_subset = object_detection_prediction[
#     object_detection_prediction["image_id"].isin(abnormal_ids)]
#
# # %% --------------------
# id_to_label = {
#     0: "aortic enlargement",
#     1: "atelectasis",
#     2: "calcification",
#     3: "cardiomegaly",
#     4: "consolidation",
#     5: "ild",
#     6: "infiltration",
#     7: "lung opacity",
#     8: "nodule/mass",
#     9: "other lesion",
#     10: "pleural effusion",
#     11: "pleural thickening",
#     12: "pneumothorax",
#     13: "pulmonary fibrosis",
#     14: "No Findings class"
# }
#
# # %% --------------------
# # normalize gt
# normalized_gt_df = normalize_bb(original_dataset, original_dataset, "transformed_height",
#                                 "transformed_width")
#
# # %% --------------------CONF + NMS
# # post process
# nms_final_df = post_process_conf_filter_nms(object_detection_prediction_subset,
#                                             obj_det_conf_thr, iou_threshold, original_image_ids)
#
# # normalize
# nms_normalized = normalize_bb(nms_final_df, original_dataset, "transformed_height",
#                               "transformed_width")
#
# # compute mAP
# print(zfturbo_compute_mAP(normalized_gt_df, nms_normalized, id_to_label, gt_class_id="class_id",
#                           pred_class_id="label"))
#
# # %% --------------------CONF + WBF
# # post process
# wbf_final_df = post_process_conf_filter_wbf(object_detection_prediction_subset,
#                                             obj_det_conf_thr, iou_threshold, original_dataset,
#                                             original_image_ids)
#
# # normalize
# wbf_normalized = normalize_bb(wbf_final_df, original_dataset, "transformed_height",
#                               "transformed_width")
#
# # compute mAP
# print(zfturbo_compute_mAP(normalized_gt_df, wbf_normalized, id_to_label, gt_class_id="class_id",
#                           pred_class_id="label"))
