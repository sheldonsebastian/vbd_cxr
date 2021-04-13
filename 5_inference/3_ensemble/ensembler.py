# %% --------------------
import pandas as pd

output_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/final_outputs"

# %% --------------------
# read resnet152 output
resnet_152 = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/3_validation_data/pipeline_10_percent/2_class_classifier/predictions/holdout_resnet152.csv")
resnet_152.columns = ['image_id', 'target_resnet_152', 'probabilities_resnet_152']

# read vgg19 output
vgg_19 = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/3_validation_data/pipeline_10_percent/2_class_classifier/predictions/holdout_vgg19.csv")
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
combined.to_csv(output_dir + "/holdout/holdout_binary_resnet152_vgg19.csv",
                index=False)
