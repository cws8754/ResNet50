import os
import sys
import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import random
from matplotlib import pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ResNet50 import ResNet, Bottleneck


T = TypeVar('T')

def parse_config(config_path: str, cls: Type[T]) -> T:
    """
    Parse a JSON config file and convert it to an instance of the given dataclass.
    
    Args:
        config_path (str): The path to the JSON config file.
        cls (Type[T]): The dataclass type to map the JSON data to.
    
    Returns:
        T: An instance of the dataclass populated with the JSON data.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)

@dataclass
class Config:
    model_name: str = field(
        default="ResNet",
        metadata={"help": "The name of the model to use."}
    )

    hidden_size: int = field(
        default=768,
        metadata={"help": "The size of the hidden layer."}
    )

    n_classes: int = field(
        default=10,
        metadata={"help": "The number of classes to classify."}
    )

    batch_size: int = field(
        default=128,
        metadata={"help": "The batch size to use."}
    )

    opt_name: str = field(
        default="adam",
        metadata={"help": "The name of the optimizer to use."}
    )

    lr: float = field(
        default=1e-3,
        metadata={"help": "The learning rate to use."}
    )

    num_epochs: int = field(
        default=100,
        metadata={"help": "The number of epochs to train for."}
    )

    early_stopping: int = field(
        default=None,
        metadata={"help": "The number of epochs to wait before early stopping."}
    )

    p: float = field(
        default=0.1,
        metadata={"help": "The dropout rate to use."}
    )

def train(model, train_loader, criterion, optimizer, device):
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.train()    
    for batch_idx, (x, y) in enumerate(train_loader):  
        x = x.to(device)
        y = y.to(device)
            
        optimizer.zero_grad()                
        y_pred, _ = model(x)  
        
        loss = criterion(y_pred, y)  
        loss.backward()        
        optimizer.step()        
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(y_pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
    epoch_loss /= len(train_loader)  
    epoch_acc = correct / total
       
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.eval()    
    with torch.no_grad():        
        for batch_idx, (x, y) in enumerate(val_loader):  
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x) 
            loss = criterion(y_pred, y)  

            epoch_loss += loss.item()
            
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
    epoch_loss /= len(val_loader)  
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

loss_train = []
loss_val = []

def main():

    config = Config()

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parse_config(sys.argv[1], Config)

    os.makedirs(f"{Config.model_name}_logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    resnet_model = ResNet(config, output_dim=config.n_classes)
    resnet_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=config.lr)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.batch_size,  
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=Config.batch_size,  
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    best_valid_loss = float('inf')  
    best_valid_acc = 0.0


    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train(resnet_model, trainloader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(resnet_model, testloader, criterion, device)
        
        if valid_acc > best_valid_acc:  
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            torch.save(resnet_model.state_dict(), f'{config.model_name}_logs/best_model.pth')  

        loss_train.append(train_loss)
        loss_val.append(valid_loss)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}%')
        print(f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:6.2f}%')
    
plt.plot(loss_train, 'yellow', label='train')
plt.plot(loss_val, 'blue', label='val')    
plt.legend()
plt.show()

if __name__ == "__main__":
    main()
