import cv2
from torch.utils.data import Dataset

CLASSTOLABEL = {"fractured": 0, "not fractured": 1}

class BoneFractureDataset(Dataset):
    
    """ A custom dataset loading class that read each and every image in
    a data directory i.e., train, test, and val.

    Args:
        Dataset (class): A module of torch dataset class.
    """
    
    def __init__(self, image_paths, transform = False):
        
        """ Initialising the class variables.

        Args:
            image_paths (string): Location to a data directory
            transform (bool): Apply transformations to an image. Defaults to False.
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        
        """ Returns the length of dataset. """
        
        return(len(self.image_paths))
    
    def __getitem__(self, idx):
        
        """ Loads each image and it's corresponding label based on an iterator idx.

        Returns:
            tensor: returns transformed image and it's corresponsing label as seperate tensors.
        """
        
        image_path = self.image_paths[idx]  # reads an image path from a list of image paths
        
        label = image_path.split('/')[-2]   # extracting label from image path - "./train/fractured/0.png"
        label = CLASSTOLABEL[label]
        
        image = cv2.imread(image_path)  # read image using opencv and the corresponding image path
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR so we need to convert them to RGB
            
        if self.transform is not None:  # apply transformations to an image if applicable
            image = self.transform(image)
            
        return image, label     # return image and label tensors