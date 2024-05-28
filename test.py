import cv2
import torch
import numpy as np

from model import CustomResnet
from utils import load_model, decodeImage
from train_helper import transform, device

from PIL import Image


MODEL_PATH = "/Users/anuragkotiyal/Desktop/Projects/Bone Fracture Detection/Models/customresnet_best_weights.pt"
LABELTOCLASS = {0: "fractured", 1: "not fractured"}
img_path = "/Users/anuragkotiyal/Desktop/Projects/Bone Fracture Detection/Dataset/test/fractured/82-rotated2-rotated3.jpg"
#img_path = "/Users/anuragkotiyal/Desktop/Projects/Bone Fracture Detection/Dataset/test/not fractured/1-rotated2-rotated1-rotated1.jpg"
def test():
    
    # image_tensor = transform(Image.open("input.jpg"))
    
    test_image = cv2.imread(img_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # test_image = np.expand_dims(test_image, axis = 0)
    test_image = transform(test_image).unsqueeze(0)
    
    model = CustomResnet(2)     # initialize model
    model = load_model(model, MODEL_PATH)       # load trained model
    
    output = model(test_image)
    _, predicted = torch.max(output.data, 1)
        
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    class_index = probabilities.argmax()
    
    if class_index == 0:
        result = { "image" : "fractured"}
        
    else:
        result = { "image" : "not fractured"}
        
    print(predicted)
    
if __name__ == "__main__":
    test()