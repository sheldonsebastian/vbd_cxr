# https://www.kaggle.com/corochann/vinbigdata-detectron2-train
# %% --------------------
import copy
import datetime
import logging
import os
import random
import time

import albumentations as A
import cv2
import detectron2.utils.comm as comm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode as CN
from detectron2.data import detection_utils as utils, MetadataCatalog, DatasetCatalog, \
    get_detection_dataset_dicts, DatasetFromList, MapDataset, build_batch_data_loader, \
    build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import TrainingSampler
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.engine.hooks import HookBase
from detectron2.structures.boxes import BoxMode
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import Visualizer

from common.detectron2_evaluator import Detectron2_ZFTurbo_mAP


# %% --------------------DATASET PREPARATION
def get_train_detectron_dataset(img_dir, annot_path, external_dir=None, external_annot_path=None):
    """
    Converts training data containing annotations into format compatible for detectron2
    :img_dir: base directory where all the images are present
    :annot_path: pandas dataframe path which contains ground truth annotations for images
    :external_dir: directory where external images are present
    :external_annot_path: pandas dataframe path which contains ground truth annotations for external
    images
    :return: returns list where each element contains dictionary as per detectron2 format
    """
    # read GT annotations
    df = pd.read_csv(annot_path)

    # get unique image_ids
    uids = sorted(list(df["image_id"].unique()))

    dataset_dicts = []

    # https://www.stackabuse.com/how-to-iterate-over-rows-in-a-pandas-dataframe/
    for uid in uids:

        img_data = df[df["image_id"] == uid]

        record = {"file_name": os.path.join(img_dir, uid + ".png"), "image_id": uid,
                  "height": 512,
                  "width": 512}

        objs = []

        for _, row in img_data[["x_min", "y_min", "x_max", "y_max", "class_id"]].iterrows():
            obj = {
                "bbox": [row.x_min, row.y_min, row.x_max, row.y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(row.class_id)
            }
            objs.append(obj)

        record["annotations"] = objs

        # append record to dataset
        dataset_dicts.append(record)

    # EXTERNAL OBJECT DETECTION DATA
    if (external_dir is not None) and (external_annot_path is not None):

        # read GT annotations
        external_df = pd.read_csv(external_annot_path)

        # only use pneumothorax = 12 and atelectasis = 1
        external_df = external_df[(external_df["class_id"] == 12) | (external_df["class_id"] == 1)]

        # get unique image_ids
        external_uids = sorted(list(external_df["image_id"].unique()))

        # https://www.stackabuse.com/how-to-iterate-over-rows-in-a-pandas-dataframe/
        for uid in external_uids:

            img_data = external_df[external_df["image_id"] == uid]

            record = {"file_name": os.path.join(external_dir, uid + ".png"), "image_id": uid,
                      "height": 512,
                      "width": 512}

            objs = []

            for _, row in img_data[["x_min", "y_min", "x_max", "y_max", "class_id"]].iterrows():
                obj = {
                    "bbox": [row.x_min, row.y_min, row.x_max, row.y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(row.class_id)
                }
                objs.append(obj)

            record["annotations"] = objs

            # append record to dataset
            dataset_dicts.append(record)

    return dataset_dicts


# %% --------------------
def get_test_detectron_dataset(img_dir, dataframe_path):
    """Converts testing data without annotations into format compatible for detectron2
    :img_dir: directory which contains test images
    :dataframe_path: path to dataframe which contains test image_ids and dimensions
    :return: returns list where each element contains dictionary as per detectron2 format without
    bounding box labels
    """
    # read annotations
    df = pd.read_csv(dataframe_path)

    # get unique image_ids
    uids = sorted(list(df["image_id"].unique()))

    dataset_dicts = []

    # https://www.stackabuse.com/how-to-iterate-over-rows-in-a-pandas-dataframe/
    for uid in uids:
        img_data = df[df["image_id"] == uid]

        record = {"file_name": os.path.join(img_dir, uid + ".png"), "image_id": uid,
                  "height": 512,
                  "width": 512}

        # append record to dataset
        dataset_dicts.append(record)

    return dataset_dicts


# %% --------------------ALBUMENTATION MAPPER USED BY DATALOADER
class AlbumentationsMapper:
    def __init__(self, cfg, is_train: bool = True):
        """
        :cfg: contains information about augmentations to apply
        :is_train: is_train = True for training only. is_train = False for validation and test
        """
        # get augmentations from cfg
        aug_kwargs = cfg.aug_kwargs

        # augmentations to apply
        aug_list = []

        # if training mode then apply augmentation, for validation and testing mode do not apply
        # augmentations
        if is_train:
            aug_list.extend([getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()])

        # create augmentation pipeline using albumentation compose
        # category_id will be added from record to transform
        self.transform = A.Compose(aug_list, bbox_params=A.BboxParams(format="pascal_voc",
                                                                      label_fields=[
                                                                          "category_ids"]))
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        print(f"Albumentations Mapper Augmentation used in {mode}: {self.transform}")

    # make the class itself callable
    def __call__(self, dataset_dict):
        """
        :dataset_dict: this contains a single record(as per dataset) which the dataloader requested
        :return: Apply augmentation and convert record(as per dataset) into format required by
        model (https://detectron2.readthedocs.io/en/v0.2.1/tutorials/models.html#model-input-format)
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        # detectron utils reads images in batch, thus shape of image = (Height x Width x Channel)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        # convert record into model input format
        prev_anno = dataset_dict["annotations"]

        bboxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)

        # make dummy category ids for indexing purposes
        category_id = np.arange(len(dataset_dict["annotations"]))

        # transform the input image
        print(dataset_dict["file_name"])
        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_id)

        # extract transformed image
        image = transformed["image"]

        # prep annotations for model input
        annos = []

        for i, j in enumerate(transformed["category_ids"]):
            # fetch old annotation using dummy category_id
            temp = prev_anno[j]

            # updating the bboxes of old annotation
            temp["bbox"] = transformed["bboxes"][i]

            annos.append(temp)

        # delete annotations as model will use "instances" instead of annotations
        dataset_dict.pop("annotations", None)

        # image.shape returns H x W x C
        # image.shape[:2] returns H x W
        image_shape = image.shape[:2]

        # transpose operation converts H x W x C --> C x H x W
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        # convert annotations to instances format
        instances = utils.annotations_to_instances(annos, image_shape)
        # delete any images which do not contain annotations
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


# %% --------------------
# https://www.kaggle.com/corochann/vinbigdata-detectron2-train
# https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    # Overwrite function
    def after_step(self):
        # after each iteration increment the iteration count
        next_iter = int(self.trainer.iter) + 1

        # check if last iteration
        is_final = next_iter == self.trainer.max_iter

        # for last iteration or after periodic evaluation steps, compute validation loss
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            # validation loss
            mean_loss = self._do_loss_eval()
            # add loss to tensorboard
            self.trainer.storage.put_scalars(validation_loss=mean_loss)
            print("validation do loss eval", mean_loss)
        else:
            pass
            # self.trainer.storage.put_scalars(timetest=11)

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        # self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        # return losses
        return mean_loss

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced


# %% --------------------
def dataset_sample_viewer(dataset_name: str, sample_size: int):
    """View sample data from Dataset"""
    # get metadata from MetadataCatalog
    train_metadata = MetadataCatalog.get(dataset_name)

    # get dataset from DatasetCatalog
    dataset = DatasetCatalog.get(dataset_name)

    for record in random.sample(dataset, sample_size):
        img = cv2.imread(record["file_name"])
        visualize = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=1)
        out = visualize.draw_dataset_dict(record)
        plt.title(record["image_id"])
        plt.imshow(out.get_image()[:, :, ::-1], cmap="gray")
        plt.tight_layout()
        plt.show()


# %% --------------------
# detectron2_codes/detectron2-windows/detectron2/data/build.py
def build_detection_train_loader_with_train_sampler(cfg, mapper, seed=42, shuffle=True):
    dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, mapper)

    logger = logging.getLogger(__name__)
    logger.info("Using training sampler TrainingSampler with shuffle=False")
    sampler = TrainingSampler(len(dataset), shuffle=shuffle, seed=seed)

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# %% --------------------
def view_sample_augmentations(dataset_name, augmentation_dictionary, n_images, n_aug, seed=42):
    """
    :augmentation_dictionary:
    Example
    {
        "HorizontalFlip": {"p": 0.5},
        "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
        "RandomBrightnessContrast": {"p": 0.5}
    }
    """
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 0

    # Batch size per iteration
    cfg.SOLVER.IMS_PER_BATCH = n_images

    # augmentations used by AlbumentationsMapper
    cfg.aug_kwargs = CN(augmentation_dictionary)

    # create dataloader using albumentationsmapper and verify augmentations
    fig, axes = plt.subplots(n_images, n_aug)

    # Ref https://github.com/facebookresearch/detectron2/blob/22b70a8078eb09da38d0fefa130d0f537562bebc/tools/visualize_data.py#L79-L88
    for i in range(n_aug):

        # create data loader
        train_vis_loader = build_detection_train_loader_with_train_sampler(cfg,
                                                                           mapper=AlbumentationsMapper(
                                                                               cfg, True),
                                                                           seed=seed,
                                                                           shuffle=False)

        # iterate through data loader
        for batch in train_vis_loader:
            for j, per_image in enumerate(batch):
                ax = axes[j, i]

                img_arr = per_image["image"].cpu().numpy().transpose((1, 2, 0))
                visualizer = Visualizer(
                    img_arr[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0
                )
                target_fields = per_image["instances"].get_fields()
                labels = [
                    MetadataCatalog.get(dataset_name).thing_classes[i] for i in
                    target_fields["gt_classes"]
                ]
                out = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                # out = visualizer.draw_dataset_dict(per_image)

                img = out.get_image()[:, :, ::-1]
                ax.imshow(img)
                ax.set_title(f"image{j}, {i}-th aug")
            break

    plt.tight_layout()
    plt.show()


# %% --------------------
def build_simple_dataloader(dataset_name: list, batch_size):
    dataset_dicts = get_detection_dataset_dicts(dataset_name)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    cfg = get_cfg()
    cfg["aug_kwargs"] = {}

    dataset = MapDataset(dataset, AlbumentationsMapper(cfg, False))

    # set the shuffle to False in debugging mode
    sampler = TrainingSampler(len(dataset), shuffle=False, seed=42)
    dataloader = build_batch_data_loader(dataset=dataset, sampler=sampler,
                                         total_batch_size=batch_size)

    return dataloader


# %% --------------------
class CustomTrainLoop(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=AlbumentationsMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            # not applying augmentation to validation data
            cfg, dataset_name, mapper=AlbumentationsMapper(cfg, False)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        conf = cfg.mAP_conf_thr
        return Detectron2_ZFTurbo_mAP(conf_thr=conf)

    def build_hooks(self):
        hooks = super(CustomTrainLoop, self).build_hooks()

        cfg = self.cfg
        if len(cfg.DATASETS.TEST) > 0:
            loss_eval_hook = LossEvalHook(cfg.TEST.EVAL_PERIOD, self.model,
                                          CustomTrainLoop.build_test_loader(cfg,
                                                                            cfg.DATASETS.TEST[0]), )
            hooks.insert(-1, loss_eval_hook)

        return hooks


# %% --------------------
# https://stackoverflow.com/questions/63578040/how-many-images-per-iteration-in-detectron2
def convert_epoch_to_max_iter(epochs, batch_size, len_dataset):
    return np.ceil((epochs * len_dataset) / batch_size)


# %% --------------------
def predict_batch(predictor: DefaultPredictor, input_img_list):
    """Function used to override the resize operation which the DefaultPredictor performs"""
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # prep input
        inputs_list = []
        for original_image in input_img_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            # Do not apply original augmentation, which is resize.
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)

        # make predictions for batch of input
        predictions = predictor.model(inputs_list)

        return predictions
