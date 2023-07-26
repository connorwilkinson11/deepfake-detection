
# importing the required libraries
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
  
# defining the Dataset class
class data_set(Dataset):
    def __init__(self):
        numbers = list(range(0, 100, 1))
        self.data = numbers
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]
  
  
dataset = data_set()
  
# implementing dataloader on the dataset and printing per batch
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
for i, batch in enumerate(dataloader):
    print(i, batch)