# Imports here
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms, models
from torch import optim
import torch.nn.functional as f
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import io
import requests
import seaborn as sb
import json
import train

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(filepath):
    
    checkpoint = torch.load(filepath)

    model_architecture = checkpoint["model_architecture"]
    
    if model_architecture == "vgg13":
        model = models.vgg13(pretrained = True)
    else:
        model = models.resnet18(pretrained = True)

    
    model.classifier = checkpoint["classifier"]
    model.epochs = checkpoint["epochs"]
    optimizer = checkpoint["optimizer"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"], strict = False)
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model        
    process = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))
                                ])
    
    image = process(image)
    np_image = np.array(image)
    np_image.transpose((2,0,1))
    return np_image



def predict_with_gpu(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    classes = []
    with Image.open(image_path) as image:
        image = process_image(image)
        
    
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    image = image.cuda()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model = model.eval()
    
    with torch.no_grad():
        result = model.forward(image)
    
#     probabilities= torch.exp(result)
    probabilities = result
    high_prob = probabilities.topk(topk)
    prob = high_prob[0].cpu().numpy()[0]
    idx = high_prob[1].cpu().numpy()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    for i in range(len(idx)):
        classes.append(idx_to_class[i])

    return prob, classes


def predict_without_gpu(image_path, model, topk = 5):

    classes = []
    
    with Image.open(image_path) as image:
        image = process_image(image)
        
    
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    image = image.cuda()
    
    model = model.eval()
    
    with torch.no_grad():
        result = model.forward(image)
    
    probabilities= torch.exp(result)
    high_prob = probabilities.topk(topk)
    prob = high_prob[0].cpu().numpy()[0]
    idx = high_prob[1].cpu().numpy()[0]
    
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    
    for i in idx:
        
        classes.append(idx_to_class[i])
    
    return prob, classes

def show_result(probs, classes):
    
    flower_names = []
    json_file = input("from which json file do you want to load the classes? Please insert the filepath of the json file: ")
    
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    
    for i in range(len(classes)):
        flower_names.append(cat_to_name.get(classes[i]))

    #sb.barplot(x = flower_names, y = probs, color = "b")
    print("These are the top5 classes: \n")
    
    for i in range(len(flower_names)):
        print("{} with the probability {}%".format(flower_names[i], probs[i]))