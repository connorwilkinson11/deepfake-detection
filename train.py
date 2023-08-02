import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
from dataset import ImageDataset

PRETRAINED_WEIGHTS_PATH = 'external_data/tf_efficientnet_b7_ns.pth'
LEARNING_RATE = 10e-2
MOMENTUM = 0.8
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 2
LABELS = "train_sample_detections/metadata.json"
def main():
    
    train = ImageDataset(LABELS, "train_sample_detections", transform=T.Resize((600,600)))
    loader = DataLoader(train)
    model = EfficientNet.from_name('efficientnet-b7')
    state = torch.load(PRETRAINED_WEIGHTS_PATH)
    
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                nesterov=True)
    optimizer.zero_grad()
    
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, NUM_CLASSES)
    sample = next(iter(loader))
    X = sample[0].float()
    pred = model(X)
    y = sample[1]
    print(pred)
    print(y)
    
    criteron = torch.nn.CrossEntropyLoss()
    loss = criteron(pred, y)
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    main()