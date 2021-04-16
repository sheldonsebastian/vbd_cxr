import os
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# ---------- 2 CLASS FILTER MODELS ----------




# ---------- OBJECT DETECTION MODELS ----------


# ----- FASTER RCNN

# ----- RETINA




# ----- YOLOV5

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

os.chdir('/Users/amnagul/Desktop/Pycharm_Projects/Flask/yolov5')

weights_dir = '/Users/amnagul/Desktop/Pycharm_Projects/Flask/yolov5/runs/train/exp/weights/best.pt'

img_id = '9a5094b2563a1ef3ff50dc5c7ff71345'
img_path = '/Users/amnagul/Desktop/Pycharm_Projects/Flask/Data/512/transformed_data/train/'
img_orig = mpimg.imread(img_path + img_id + '.jpeg')
height, width = img_orig.shape

# making detection/inference using yolov5
os.system(f'python detect.py --weights {weights_dir} --source {img_path+img_id}.jpeg --img 640 --conf 0.01 --iou 0.4  --save-conf --save-txt --exist-ok')


pred_path = '/Users/amnagul/Desktop/Pycharm_Projects/Flask/yolov5/runs/detect/exp/'

NORMAL = False

try:
    pred_labels_yolo = pred_path + 'labels/' + img_id + '.txt'  # this file will not be generated if model does not predict any BB
    f = open(pred_labels_yolo, 'r')
    img_pred = mpimg.imread(pred_path + img_id + '.jpeg')
    imgplot = plt.imshow(img_pred)
    plt.show()
except FileNotFoundError:
    print("No abnormality found!")
    NORMAL = True




# ---------- ENSEMBLE & POST PROCESSING ----------

if NORMAL:
    bboxes = ['14', '1', '0', '0', '1', '1']
else:
    # convert predicted BB coordinates from yolo to voc
    f = open(pred_labels_yolo, 'r')
    data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)
    # rearranging predictions as class_id, conf, BB_yolo
    data = data[:, [0, 5, 1, 2, 3, 4]]


    def yolo2voc(image_height, image_width, bboxes):
        """
        yolo => [xmid, ymid, w, h] (normalized)
        voc  => [x1, y1, x2, y1]

        """
        bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

        bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
        bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

        return bboxes

    # Predictions in VOC format resized to original size
    bboxes = list(np.round(np.concatenate((data[:, :2], np.round(yolo2voc(height, width, data[:, 2:]))), axis =1).reshape(-1), 1).astype(str))




# ---------- DISPLAY FINAL PREDICTED BBs ----------


