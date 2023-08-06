import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
from dataset import ImageDataset
import numpy as np
from utils import log_training

PRETRAINED_WEIGHTS_PATH = 'external_data/tf_efficientnet_b7_ns.pth'
LEARNING_RATE = 10e-2
MOMENTUM = 0.8
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 2
LABELS = "train_sample_detections/metadata.json"
NUM_EPOCHS = 2


def train_epoch(data_loader, model, criterion, optimizer):
    for idx, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(X.float())
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def predictions(logits):
    predictions, index = torch.max(logits, 1)
    return index

def evaluate_epoch(data_loader, model, criterion):
        y_true, y_pred, running_loss = [], [], []
        for idx, (X, y) in enumerate(data_loader):
            with torch.no_grad():
                output = model(X.float())
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                total += 1
                correct += (predicted == y)
                running_loss.append(criterion(output, y).item())
        
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        stats = [loss, acc]
        log_training(stats)
        return acc, loss
    

def main():
    
    train = ImageDataset(LABELS, "train_sample_detections", transform=T.Resize((600,600)))
    train_loader = DataLoader(train)
    model = EfficientNet.from_name('efficientnet-b7')
    state = torch.load(PRETRAINED_WEIGHTS_PATH)
    
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                nesterov=True)
    optimizer.zero_grad()
    
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
            print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
            print('-' * 10)
            train_epoch(train_loader, model, criterion, optimizer)
            evaluate_epoch(train_loader, model, criterion)
            
if __name__ == "__main__":
    main()