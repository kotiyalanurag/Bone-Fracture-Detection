import cv2
import torch

from model import CustomResnet
from utils import load_model, decodeImage
from train_helper import transform, device

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "/app/Models/customresnet_best_weights.pt"

@app.route("/", methods = ["GET"])
def home():
    
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    
    image = request.json['image']   # load image
    decodeImage(image, "input.jpg")     # decode image
    
    model = CustomResnet(2)     # initialize model
    model = load_model(model, MODEL_PATH)       # load trained model
    model.to(device)
    
    test_image = cv2.imread("input.jpg")
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = transform(test_image).unsqueeze(0)
    
    model = CustomResnet(2)     # initialize model
    model = load_model(model, MODEL_PATH)       # load trained model
    
    output = model(test_image)  # get predictions
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    class_index = probabilities.argmax()    # get class index i.e., 0 or 1
    
    if class_index == 0:
        result = [{ "image" : "fractured"}]
        
    else:
        result = [{ "image" : "not fractured"}]
    
    return jsonify(result)