import os
import random
import glob

from datetime import datetime
from pandas.core.common import flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# utility functions to display current time or timestamps when executing code

def time():
    
    """ Returns the current time at execution in hour, minute, seconds format. """
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    return(current_time)

def timestamp():
    
    """ Returns the current timestamp at execution in year, month, day  hour, minute, seconds format. """
    
    return('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))

# utility function to display images from the dataset

def show_image_samples(location):
    
    """ A function that takes in the path of training, validation, or testing directory
    and displays 10 random images from both categories i.e., 5 fratured and 5 not fractured.

    Args:
        location (string): path to train, test, or validation dataset
    """
    labels = os.listdir(location)
    
    for label in labels:    # iterate through labels
        
        if label != '.DS_Store':
            
            print(f"Class: {label}")    # print label
            
            label_path = os.path.join(location, label)
            
            files = random.sample(os.listdir(label_path), 5)    # sample out 5 random images for each label
            c, r = 5, 2
            
            figsize = (16, 8)
            plt.figure(figsize = figsize)
            
            for i, file in enumerate(files):    # create sub-plots and display images
                
                plt.subplot(r, c, i+1)
                
                filepath = os.path.join(label_path, file)
                
                file = plt.imread(filepath)
                
                plt.imshow(file, aspect = None)
                plt.xticks([])
                plt.yticks([])
            
            plt.tight_layout()
            plt.show()
 
 # utility function to create file paths for each image in our dataset
            
def create_files(filepath):
    
    """ Creates a file path for each image in train, test, and validation dataset.

    Args:
        filepath (string): Location to the data directory

    Returns:
        list: A list containing file paths to each image in a data directory
    """
    
    image_path = []
    classes = []
    
    for data in glob.glob(filepath + '/*'): # save each image path in a list
        image_path.append(glob.glob(data + '/*'))
    
    image_path = list(flatten(image_path))
    
    for img in image_path:  # save the class of every image in a seperate list
        classes.append(img.split('/')[-2])
        
    return image_path   # a list of all image paths in a data directory

# utility function to create dataloaders using torch dataloader

def create_data(dataset, batch_size):
    
    """ A dataloader that creates an iterable using which we can train and evaluate our model.

    Args:
        dataset (torch.Dataset): an instance of torch.Dataset class
        batch_size (int): batch size of data i.e., 4, 8, 16, or 32

    Returns:
        torch.DataLoader: returns an iterable instance of torch.DataLoader (that contains image and label)
    """
    
    return DataLoader(dataset, batch_size, shuffle = True)

# utility functions for training, validating, and evaluating our model

def train_epoch(model, device, train_data, train_image_paths, optimizer, loss_fn):
    
    """ A single training step for our model with training data

    Args:
        model (torchvision.models): a torchvision model
        device (device): mps or cpu
        train_data (torch.utils.data.DataLoader): an iterable dataloader that contains image and label tensors
        train_image_paths (list): list of image paths to all training images
        optimizer (torch.optim): a torch optimizer like Adam
        loss_fn (torch.nn): a torch loss function like categorical crossentropy

    Returns:
        tensor, float: returns avg train loss and accuracy for a single epoch
    """
    
    train_loss = 0.0
    correct = 0
    
    for i, batch in enumerate(train_data):  # iterating over a training batch
        
        images, labels = batch  # loading image, label from a batch
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()   # reset model gradients to zero
        
        outputs = model(images) # pass image tensors through model
        _, predicted = torch.max(outputs.data, 1)   # get predictions for image tensors from model
        
        loss = loss_fn(outputs, labels) # calculate loss between predictions and labels
        loss.backward() # backward step on loss function
        
        optimizer.step()
        
        train_loss += loss.item()
        avg_train_loss = train_loss/(i+1)
        
        correct += (predicted == labels).float().sum() 
        
    train_accuracy = 100 * correct / len(train_image_paths)
    
    return avg_train_loss, train_accuracy


def val_epoch(model, device, val_data, val_image_paths, loss_fn):
    
    """ A single evaluation step for our model with validation data

    Args:
        model (torchvision.models): a torchvision model
        device (device): mps or cpu
        val_data (torch.utils.data.DataLoader): an iterable dataloader that contains image and label tensors
        val_image_paths (list): list of image paths to all validation images
        optimizer (torch.optim): a torch optimizer like Adam
        loss_fn (torch.nn): a torch loss function like categorical crossentropy

    Returns:
        tensor, float: returns avg validation loss and accuracy for a single epoch
    """
    
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():   # in this mode the model can't compute gradients (only used for evaluating models)
        
        for i , batch in enumerate(val_data):   # iterating over a validation batch
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
            correct += (predicted == labels).float().sum()
    
    avg_val_loss = val_loss/(i+1)
    val_accuracy = 100 * correct / len(val_image_paths)
    
    return avg_val_loss, val_accuracy

def test_epoch(model, device, test_data, test_image_paths, loss_fn):
   
    """ A single evaluation step for our model with test data

    Args:
        model (torchvision.models): a torchvision model
        device (device): mps or cpu
        test_data (torch.utils.data.DataLoader): an iterable dataloader that contains image and label tensors
        test_image_paths (list): list of image paths to all test images
        optimizer (torch.optim): a torch optimizer like Adam
        loss_fn (torch.nn): a torch loss function like categorical crossentropy

    Returns:
        tensor, float: returns avg test loss and accuracy for a single epoch
    """
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad(): # in this mode the model can't compute gradients (only used for evaluating models)
        
        for i, batch in enumerate(test_data):   # iterating over a test batch
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            correct += (predicted == labels).float().sum()
    
    avg_test_loss = test_loss/(i+1)
    test_accuracy = 100 * correct / len(test_image_paths)
    
    return avg_test_loss, test_accuracy

# utility functions to save and load pytorch models

def save_model(model, path):
    
    """ A function to save a torch model in a specified path.

    Args:
        model (torchvision.models): a pytorch model
        path (string): a path to save the pytorch model
    """
    
    torch.save(model.state_dict(), f"{path}")
    
def load_model(model, path):
    
    """ A function to load a torch model from a specified path

    Args:
        model (torchvision.models): a pytorch model
        path (string): a path to load the pytorch model

    Returns:
        torchvision.models: a pytorch model
    """
    
    model.load_state_dict(torch.load(path))
    
    return model

# utility function to get predictions from a torch model

def get_predictions(model, device, test_data):
    
    """ A function to get predictions for a batch of test data from our model.

    Args:
        model (torchvision.models): a pytorch model
        device (device): mps or cpu
        test_data (torch.utils.data.DataLoader): an iterable batch of image and label tensors

    Returns:
        tensor: prediction and label tensors
    """
    
    model.eval()    # setting model in eval mode
    
    label = []
    prediction = []
    
    with torch.no_grad():   # model doesn't compute gradients in this mode
        
        for i, batch in enumerate(test_data):   # a batch of test images and labels
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            prediction.extend(predicted)
            label.extend(labels)
    
    return prediction, label

# utility function to visualise model performance

def plot_confusion_matrix(labels, predictions):
    """ A function to plot a confusion matrix given model predictions and true labels of images

    Args:
        labels (list): list of true labels
        predictions (list): list of corresponsing model predictions
    """
    cm = confusion_matrix(labels, predictions, labels = [0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['fractured', 'not fractured'],)                                                      
    disp.plot()
    plt.show()
    
def plot_model_performance(train_loss, train_acc, val_loss, val_acc, name = "Custom Resnet"):
    
    """ A function to plot the train/val accuracies and loss over different epochs during model training.

    Args:
        train_loss (list): a list of train loss over the epochs
        train_acc (tensor): a list of tensors of train accuracy over the epochs
        val_loss (list):  a list of val loss over the epochs
        val_acc (tensor): a list of tensors of val accuracy over the epochs
        name (str, optional): A name for the figure. Defaults to "Custom Resnet".
    """
    
    legends = ['train', 'validation']
    
    plt.figure(figsize = (20, 5))
    
    train_acc = [t.cpu().numpy().tolist() for t in train_acc]   # can't plot tensors directly - need to move them to cpu first, then convert to numpy and to a list
    val_acc = [v.cpu().numpy().tolist() for v in val_acc]
    
    plt.subplot(121)
    plt.plot(train_acc, marker = 'x')
    plt.plot(val_acc, marker = 'o')
    
    plt.title(name + '\n' + timestamp(), fontsize=18)
    plt.xlabel('Epochs', fontsize = 15)
    plt.ylabel('Accuracy', fontsize = 15)
    plt.legend(legends, loc = 'upper left')
    plt.grid()
    
    train_loss = train_loss
    val_loss = val_loss
    
    plt.subplot(122)
    plt.plot(train_loss, marker = 'x')
    plt.plot(val_loss, marker = 'o')
    
    plt.title(name + '\n' + timestamp(), fontsize=18)
    plt.xlabel('Epochs', fontsize = 15)
    plt.ylabel('Loss', fontsize = 15)
    plt.legend(legends, loc = 'upper left')
    plt.grid()
    plt.show()