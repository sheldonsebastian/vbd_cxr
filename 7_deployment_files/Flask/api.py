from flask import Flask
from flask import render_template
from flask import request
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

app = Flask(__name__)
# location to save the uploaded image before passing it to model
UPLOAD_FOLDER = '/Users/amnagul/Desktop/Pycharm_Projects/Flask/Data/'

def yolov5_predictions(image_path):
    os.chdir('/Users/amnagul/Desktop/Pycharm_Projects/Flask/yolov5')
    weights_dir = '/Users/amnagul/Desktop/Pycharm_Projects/Flask/yolov5/runs/train/exp/weights/best.pt'

    img_orig = mpimg.imread(image_path)
    height, width = img_orig.shape[0:2]

    # making detection/inference using yolov5
    os.system(f'python detect.py --weights {weights_dir} --source {image_path} --img 640 --conf 0.01 --iou 0.4  --save-conf --save-txt --exist-ok')

    pred_path = '/Users/amnagul/Desktop/Pycharm_Projects/Flask/yolov5/runs/detect/exp/'

    NORMAL = False
    try:
        pred_labels_yolo = pred_path + 'labels/' + (image_path.rsplit('/', 1)[-1]).split('.')[0] + '.txt'  # this file will not be generated if model does not predict any BB
        f = open(pred_labels_yolo, 'r')
        img_pred = mpimg.imread(pred_path + image_path.rsplit('/', 1)[-1])
        imgplot = plt.imshow(img_pred)
        plt.show()

    except FileNotFoundError:
        print("No abnormality found!")
        NORMAL = True

    pred_img_path = pred_path + image_path.rsplit('/', 1)[-1]

    # copying predicted image BB to static folder for easy retrieval (after.html page)
    os.system(f'cp {pred_img_path} /Users/amnagul/Desktop/Pycharm_Projects/Flask/static/images/')

    return image_path.rsplit('/', 1)[-1]    # image_name i.e. abc.jpg

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_file.save(f"{UPLOAD_FOLDER + image_file.filename}")
            pred_img_path = yolov5_predictions(f"{UPLOAD_FOLDER + image_file.filename}")
            return render_template("after.html", pred_img_path = pred_img_path)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(port=12000, debug=True)

