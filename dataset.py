import os
import pandas as pd
import torch
import glob
import cv2
from torch.utils.data import Dataset
from torchvision.io import read_video

VIDEOS_PATH = "train_sample_videos/"
IMAGES_PATH = "train_sample_detections/"
ANNOTATIONS_PATH = "train_sample_videos/metadata.json"

class VideoDataset(Dataset):
    def __init__(self, annotations_file, videos_dir, transform=None, target_transform=None):
        self.video_labels = pd.read_json(annotations_file).T
        self.videos_dir = videos_dir
        self.video_paths = []
        for path in glob.glob(os.path.join(self.videos_dir, '*.mp4')):
            self.video_paths.append(path)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        item = {
            "index" : idx,
            "path"  : video_path,
        }       

        return item

class ImageDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        self.frame_labels = pd.read_json(annotations_file).T
        self.images_dir = images_dir
        self.image_paths = []
        for path in glob.glob(os.path.join(self.images_dir, '*.jpg')):
            self.image_paths.append(path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        name = image_path.split("/")[1]
        label_dict = {
            "REAL" : 0,
            "FAKE" : 1
        }
        label = label_dict[self.frame_labels.loc[name][0]]
        image = cv2.imread(image_path)
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1)
        transformed_image = self.transform(image_tensor)      
        return transformed_image, label
    
def main():
    dataset = VideoDataset(ANNOTATIONS_PATH, VIDEOS_PATH)
    print(dataset.video_labels)
    print(dataset.__getitem__(0))

if __name__ == "__main__":
    main()
