# %% --------------------
import pandas as pd

output_dir = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/final_outputs"

# %% --------------------
# read resnet50 output
resnet_50 = pd.read_csv(output_dir + "/test_predictions/test_2_class_resnet50.csv")
resnet_50.columns = ['image_id', 'target_resnet_50', 'probabilities_resnet_50']

# read resnet152 output
resnet_152 = pd.read_csv(output_dir + "/test_predictions/test_2_class_resnet152.csv")
resnet_152.columns = ['image_id', 'target_resnet_152', 'probabilities_resnet_152']

# read vgg19 output
vgg_19 = pd.read_csv(output_dir + "/test_predictions/test_2_class_vgg19.csv")
vgg_19.columns = ['image_id', 'target_vgg_19', 'probabilities_vgg_19']

# %% --------------------
# join dataframes using image_id as key
combined = resnet_50.join(resnet_152.set_index("image_id"), on="image_id")
combined = combined.join(vgg_19.set_index("image_id"), on="image_id")

combined = combined[
    ["image_id", "probabilities_resnet_50", "probabilities_resnet_152", "probabilities_vgg_19"]]

# taking average of all models
combined["probabilities"] = (combined["probabilities_resnet_50"] + combined[
    "probabilities_resnet_152"] + combined["probabilities_vgg_19"]) / 3

# %% --------------------
combined = combined[["image_id", "probabilities"]]

# %% --------------------
combined.to_csv(output_dir + "/test_predictions/test_2_class_ensembled.csv", index=False)
