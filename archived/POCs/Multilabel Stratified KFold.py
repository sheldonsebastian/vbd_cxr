# %% --------------------
# https://github.com/trent-b/iterative-stratification#multilabelstratifiedkfold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# %% --------------------

img_id_arr = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
class_id = np.array([0, 0, 1, 2, 3, 2, 4, 5, 6, 0])
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# %% --------------------
df = pd.DataFrame(
    {"img_id": img_id_arr, "class_id": class_id, "x1": x1, "x2": x2, "y1": y1, "y2": y2})

# %% --------------------
df["fold"] = -1

# %% --------------------
unique_image_ids = df["img_id"].unique()
unique_classes = df["class_id"].unique()

# %% --------------------
one_hot_labels = []
for img_id in unique_image_ids:
    classes = df[df["img_id"] == img_id]["class_id"].values
    x = np.eye(len(unique_classes))[classes.astype(int)].sum(0)
    one_hot_labels.append(x)

one_hot_labels = np.array(one_hot_labels)

# %% --------------------
n_splits = 3
# mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
mskf = MultilabelStratifiedShuffleSplit(n_splits=n_splits, train_size=0.5, test_size=0.5,
                                        random_state=2021)

# %% --------------------
train_df = pd.DataFrame()
val_df = pd.DataFrame()

# %% --------------------
# X is unique image_id
for fold, (train_index, val_index) in enumerate(mskf.split(unique_image_ids, one_hot_labels)):
    train_data = df[df["img_id"].isin(unique_image_ids[train_index])].copy(deep=True)
    val_data = df[df["img_id"].isin(unique_image_ids[val_index])].copy(deep=True)

    train_data["fold"] = fold
    val_data["fold"] = fold

    train_df = train_df.append(train_data, ignore_index=True)
    val_df = val_df.append(val_data, ignore_index=True)

# %% --------------------
# Visualize the splits
for i in range(n_splits):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    for dataframe, type_of, ax in zip([train_df, val_df], ["Train", "Validation"], [ax1, ax2]):
        # check distribution of data
        dataframe[dataframe["fold"] == i]['class_id'].value_counts().reset_index().sort_values(
            ["index"])["class_id"].plot(kind='bar', color="blue", ax=ax)
        ax.set_title(f"{type_of} Fold {i}")

    plt.show()
