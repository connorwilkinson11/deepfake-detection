import torch
from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet
from dataset import ImageDataset

PRETRAINED_WEIGHTS_PATH = 'external_data/tf_efficientnet_b7_ns.pth'
LEARNING_RATE = 10e-2
MOMENTUM = 0.8
WEIGHT_DECAY = 1e-2
LABELS = "train_sample_detections/metadata.json"
def main():
    
    train = ImageDataset(LABELS, "train_sample_detections")
    loader = DataLoader(train)

    model = EfficientNet.from_name('efficientnet-b7')
    state = torch.load(PRETRAINED_WEIGHTS_PATH)
    
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                        nesterov=True)
    optimizer.zero_grad()

    l = [module for module in model.modules()]
    print(l)
    sample = next(iter(loader))
    image = sample[0].float()
    X = image.permute(0, 3, 1, 2)
    print(X.shape)
    pred = model(X)
    y = sample[1]
    
    criteron = torch.nn.CrossEntropyLoss()
    loss = criteron(pred, y)
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    main()