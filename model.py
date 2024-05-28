import torch.nn as nn
import torchvision.models as models

class CustomResnet(nn.Module):
    
    """ A class to create a custom resnet model with pre-trained weights
    from imagenet1k. The resnet will act as a feature extractor for our
    dataset. For classification of images, a new classification head is 
    added on top of the pre-trained resnet50.  

    Args:
        nn (class): an extension of the torch.nn module
    """
    def __init__(self, num_classes, train_basemodel = False):
        
        """ Just initialising the base model as resnet50, accepting number of classes for
        classification head, and training basemodel or just the model classifier.

        Args:
            num_classes (int): number of classes in dataset i.e., 2 (fratured, not fractured)
            train_basemodel (bool, optional): Using this we can fine-tune the whole model
            or just fine-tune the new classification head. Defaults to False means don't train
            base-model i.e., resnet50.
        """
        super(CustomResnet, self).__init__()
        self.num_classes = num_classes
        self.train_basemodel = train_basemodel
        self.resnet = models.resnet50(weights = "IMAGENET1K_V1")
        self.classifier = nn.Sequential(nn.Linear(1000, 128),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(128, 2),
                                        nn.ReLU())
        
    def forward(self, x):
        
        """ Defines the forward pass of entire model i.e., resnet 50 + classification head.

        Args:
            x (tensor): feature representation of a single input data point.

        Returns:
            tensor: Output tensor for the respective input feature representation.
        """
        x = self.resnet(x)  # passing x through resnet and saving output in x
        x = self.classifier(x)  # passing output of resnet through classifier
        
        if (self.train_basemodel):  # fine-tune the entire model i.e., resnet + classifier
            for name, param in self.resnet.named_parameters():
                param.requires_grad = True
        else:   # fine-tune only the classifier
            for name, param in self.resnet.named_parameters():
                param.requires_grad = False
            
        return x    # return the output tensor
    
if __name__ == "__main__":
    
    model = CustomResnet(2)