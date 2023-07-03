import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    '''
    Custom dataset class for the lazy loading of the data

    - type: Generated or Real Life

    - set: train, validation or test

    - balance: If true, applies oversampling AND data augmentation to the minority classes

    - filter_array: If not empty, only the images with the indexes in the array will be retained
    '''
    def __init__(self, set_type, set_name, balance, filter_array):

        if balance:
            # Transform to apply to the minibatches for data augmentation
            # Define the transformation to apply
            # Transformations: Random horizontal and vertical flips, halving and doubling the brightness
            # This should improve the prediction accuracy
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((100, 100), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=[0.75, 1.25])], p=0.5)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((100, 100), antialias=True)
            ])
            
        # Define the data_path depending on the type of the dataset and the set
        # Note the data path is relative to the file that is calling the dataset class
        # The calls are made in MODEL/MODEL_NAME/Model.py
        if set_type == "Generated":
            self.root_path = "../../Datasets/Processed Generated Data.dat/"
            if set_name == "train":
                full_data_path = "../../Datasets/Processed Generated Data.dat/train_full_generated_data.json"
            elif set_name == "validation":
                full_data_path = "../../Datasets/Processed Generated Data.dat/validation_full_generated_data.json"
            else:
                full_data_path = "../../Datasets/Processed Generated Data.dat/test_full_generated_data.json"
        else:
            self.root_path = "../../Datasets/Real Life Data/"
            if set_name == "train":
                full_data_path = "../../Datasets/Processed Real Data.dat/train_full_real_life_data.json"
            elif set_name == "validation":
                full_data_path = "../../Datasets/Processed Real Data.dat/validation_full_real_life_data.json"
            else:
                full_data_path = "../../Datasets/Processed Real Data.dat/test_full_real_life_data.json"
        
        # Load the JSON file
        with open(full_data_path, "r") as file:
            full_data = json.load(file)
        
        # Get the images name form the JSON file 
        self.images = np.array(full_data[set_name])
        
        # Get the labels from the JSON file
        self.labels = full_data["label"]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # If the retain array is not empty, only retain the images with the indexes in the array
        if len(filter_array) > 0:
            self.images = self.images[filter_array]
            self.labels = self.labels[filter_array]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.images[idx])

        image = Image.open(img_path)
        image = self.transform(image)
        
        # Convert the label to int instead of float
        label = self.labels[idx]
        
        return image, label

# We can now proceed to defining a function that creates a data loader for both datasets, oversampling the minority classes and applying horizontal flip and blur transformations:


def get_loader(set_type, set_name, batch_size, balance=True, filter_array=[]):

    # Check data validity
    if set_type not in ["Generated", "Real"]:
        raise ValueError("Dataset not valid")
    if set_name not in ["train", "validation", "test"]:
        raise ValueError("Set not valid")

    # Create the dataset
    dataset = CustomDataset(set_type, set_name, balance, filter_array)

    # If we are using a weighted random sampler (balanced case), we can retrieve the class distribution of the set at hand
    if balance:

        if(type == "Generated"):
            # Load the class distribution from the JSON file
            with open("../../Datasets PreProcessing/Data Generation/class_distribution.json", "r") as file:
                class_proportions = np.array(
                    list(json.load(file)[set_name].values()))
            # Compute the percentages
            class_proportions = class_proportions / np.sum(class_proportions)

        else:
            # Load the class distribution from the JSON file
            with open("../../Datasets PreProcessing/Real life data/class_distribution.json", "r") as file:
                class_proportions = np.array(
                    list(json.load(file)[set_name].values()))
            # Compute the percentages
            class_proportions = class_proportions / np.sum(class_proportions)

        # Define the sampler using class distributions to oversample the minority classes
        class_weights = 1. / torch.tensor(class_proportions, dtype=torch.float) # The weights of the classes
        sample_weights = class_weights[dataset.labels] # Assign each label its corresponding weight
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(dataset), replacement=True)

    else:
        sampler = None
        
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
