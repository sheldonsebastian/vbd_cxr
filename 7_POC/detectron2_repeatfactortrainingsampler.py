from detectron2.data.samplers import RepeatFactorTrainingSampler

# %% --------------------
# https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.samplers.RepeatFactorTrainingSampler
# https://github.com/facebookresearch/detectron2/issues/2139
dataset_dicts_1 = [
    {"annotations": [{"category_id": 0}, {"category_id": 1}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}, {"category_id": 2}]},
    {"annotations": [{"category_id": 0}, {"category_id": 2}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},
    {"annotations": [{"category_id": 0}]},

]
sampler = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset_dicts_1, 3)
# Inherit the Sampler, based on the threshold to reduce the problem of sample imbalance,
# categories with fewer samples are more likely to be sampled, which is suitable for the LVIS
# data set
print(sampler)
