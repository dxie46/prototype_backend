import platform
from flask import Flask, request, jsonify
from PIL import Image
import pickle
import os
import pathlib
from fastai.vision.all import *
from main import label_func, train

app = Flask(__name__)

@app.route('/eye-pressure/predict', methods=["POST"])
def predict():

    file = request.files['image']
    image = PILImage.create(file.stream)

    # if not os.path.exists('./transfer_learn_fastai.pkl'):
    #     print('model doesnt exist, creating it')
    #     train()
    if file == None:
        return jsonify({"status": "error", "result": "no image found", "probability": "0"})
    # make prediction
    print(os.getcwd())
    plt = platform.system()
    pathlib.WindowsPath = pathlib.PosixPath # needed for heroku
    learn = load_learner('./transfer_learn_fastai.pkl')
    result = learn.predict(image)
    if (result[0] < 30):
        return jsonify({"status": "warning", "result": "probability < 30", "probability": str(result[2].numpy()[0])})
    return jsonify({"status": "ok", "result": result[0], "probability": str(result[2].numpy()[0])})