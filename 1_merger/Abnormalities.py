# %% --------------------
import pandas as pd

# %% --------------------
train_bb = pd.read_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/1_merger/fused_train_0_6.csv")

# %% --------------------
train_bb_abnormalities = train_bb[train_bb["class_id"] != 14.0]

# %% --------------------
len(train_bb_abnormalities)  # 25959

# %% --------------------
train_bb_abnormalities.reset_index(drop=True).to_csv(
    "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm Workspace/vbd_cxr/1_merger/abnormalities_bb.csv",
    index=False)

# %% --------------------
