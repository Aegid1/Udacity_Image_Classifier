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
import predict

def train_model(model_architecture, lr, epochs, gpu, h1, h2, topk):
    
    data_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                                        ])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                                        ])

# TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder("./flowers", transform = data_transforms)
    image_train_dataset = datasets.ImageFolder("./flowers/train", transform = train_transforms)
    image_test_dataset = datasets.ImageFolder("./flowers/test", transform = data_transforms)
    image_validation_dataset = datasets.ImageFolder("./flowers/valid", transform = data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle = True)
    trainloader = torch.utils.data.DataLoader(image_train_dataset, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(image_test_dataset, batch_size=32, shuffle = True)
    validationloader = torch.utils.data.DataLoader(image_validation_dataset, batch_size=32, shuffle = True)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
#####################
    if model_architecture == "vgg13":
        model = models.vgg13(pretrained = True)
    if model_architecture == "vgg19":
        model = models.vgg19(pretrained = True)
    
    
    for param in model.parameters():
        param.requires_grad = False
################## 
    if gpu == "yes":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Build and train your network
    classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(25088, h1)),
                                            ("relu1", nn.ReLU()),
                                            ("drop1", nn.Dropout(p = .2)),
                                            ("fc2", nn.Linear(h1, h2)),
                                            ("relu2", nn.ReLU()),
                                            ("drop2", nn.Dropout(p = .2)),
                                            ("fc3", nn.Linear(h2, 102)),
                                            ("output", nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier
    model.to(device)

    images, labels = next(iter(testloader))
    criterion = nn.NLLLoss()
###########################
    optimizer = optim.SGD(model.classifier.parameters(), lr)
###################
    
    for e in range(epochs):
    
        running_loss = 0
        for images, labels in trainloader:
            
            if gpu == "yes":
                images = images.to(device)
                labels = labels.to(device)
        
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()
        
        else:
        
            testloss = 0
            accuracy = 0
    
            with torch.no_grad():
            
                model.eval()
            
                for images, labels in testloader:
                    
                    if gpu == "yes":
                        images = images.to(device)
                        labels = labels.to(device)
        
                    log_ps = model.forward(images)
                    testloss = testloss + criterion(log_ps, labels)
                    probabilities = torch.exp(log_ps)
                    top_p, top_class = probabilities.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = accuracy + torch.mean(equals.type(torch.FloatTensor))
        
    
            model.train()

            print("Epoch = {}".format(e+1))
            print("runningloss = {}".format(running_loss/len(trainloader)*100))
            print("testloss = {}".format(testloss/len(testloader)))
            print("accuracy = {}%".format(accuracy/len(testloader)*100))

        
    #Saving Checkpoint
# def save_checkpoint():    
    model.class_to_idx = image_train_dataset.class_to_idx

    checkpoint = {"model_architecture": model_architecture,
                "optimizer": optimizer,
                "classifier": model.classifier,
                "state_dict": model.state_dict(),
                "epochs": epochs,
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": model.class_to_idx,
              
                }

    torch.save(checkpoint, "checkpoint.pth")





