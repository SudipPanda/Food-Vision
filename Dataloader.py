import os
from torchvision import datasets , transforms
import torch
from torch.utils.data import DataLoader

def create_dataloader(
    train_dir : str ,
    test_dir:str ,
    transformer: transforms.Compose,
    batch_size : int ,
    workers : int
):
    # creating the dataset for both the train and the test set here
    train_data = datasets.ImageFolder(train_dir , transform=transformer)
    test_data = datasets.ImageFolder(test_dir , transform=transformer)

    # creatinf the dataloader
    train_dataloader = DataLoader( train_data , batch_size = batch_size ,num_workers=workers , shuffle = True)
    test_dataloader = DataLoader(test_data , batch_size=batch_size ,num_workers=workers , shuffle = False)

    class_name = train_data.classes

    return train_dataloader , test_dataloader , class_name
