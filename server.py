from flask import Flask, request, jsonify
from PIL import Image
import pickle
import os
from fastai.vision.all import *
from main import label_func, train

app = Flask(__name__)

@app.route('/eye-pressure/predict', methods=["POST"])
def predict():

    file = request.files['image']
    image = PILImage.create(file.stream)

    if not os.path.exists('./transfer_learn_fastai.pkl'):
        print('model doesnt exist, creating it')
        train()
    
    # make prediction
    print(os.getcwd())
    learn = load_learner('./transfer_learn_fastai.pkl')
    result = learn.predict(image)
    return jsonify({"result": result[0], "probability": str(result[2].numpy()[0])})