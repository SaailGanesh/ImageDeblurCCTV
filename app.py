from __future__ import print_function
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")
import os
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import shutil
import time
import re
import base64
import cv2
import numpy as np
import pandas as pd


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["IMAGE_UPLOADS"] = "./static/upload_video"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF", "MP4"]
app.config["RESULT"] = "./static/resize_img"
app.config["FINISH"] = "./static/finish"

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print("image", image)
            print("image name", image.filename)

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                print("Image saved")
                return redirect(url_for("predict",filename=filename))

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("index.html")


@app.route("/predict/<filename>", methods=["GET"])
def predict(filename):
    filname1=filename.split('.')
    print(filname1[1])
    if filname1[1] == 'mp4':
        print("\n")
        print("MP4")
        print("MP4")
        cmd_line1 = "python katna.py "+ str(filename)
        os.system(cmd_line1)
        filename = os.path.splitext(filename)[0] + "_0.jpeg"
    else:
        print(filname1[1])
        print(filname1[1])
        print(filname1[1])
        shutil.move(os.path.join(app.config["IMAGE_UPLOADS"], filename), os.path.join(app.config["RESULT"], filename))
    print(filename)
    cmd_re="python resizin.py "+str(filename)
    os.system(cmd_re)
    cmd_line="python Test.py "+str(filename)
    os.system(cmd_line)
    return render_template('result.html', filename=filename)

@app.route("/npr/<filename>", methods=["GET"])
def recognize(filename):
    cmd_npd="python npd.py --input ./static/deblur_result/"+str(filename)+ " --output ./static/detection_result/"+str(filename)+" --size 608"
    print("NPD")
    print("NPD")
    print("NPD")
    print("NPD")
    print("NPD")
    print("NPD")
    print(cmd_npd)
    os.system(cmd_npd)
    print("filename", filename)
    print("Hello World")
    cmd_nr="python img2txt.py "+str(filename)
    os.system(cmd_nr)
    cmd_np="python draw.py "+str(filename)
    os.system(cmd_np)
    time.sleep(7)
    return render_template('result1.html', filename=filename)

@app.route("/about")
def about():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug= True)