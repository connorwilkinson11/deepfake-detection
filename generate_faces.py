from dataset import VideoDataset, ANNOTATIONS_PATH, VIDEOS_PATH
from torch.utils.data import DataLoader
from tqdm import tqdm
from face_detection.face_ssd_infer import SSD
from face_detection.utils import vis_detections
import torch
import os
import cv2
import numpy as np

FACE_DETECTION_PATH = "external_data/WIDERFace_DSFD_RES152.pth"
DETECTIONS_PATH = "train_sample_detections/"
device = torch.device("cpu")
CONF_THRESH = 0.3
TARGET_SIZE = (512, 512)

net = SSD("test")

net.load_state_dict(torch.load(FACE_DETECTION_PATH, map_location="cpu"))
net.to(device).eval();

def main():
    videos = VideoDataset(ANNOTATIONS_PATH, VIDEOS_PATH)
    loader = DataLoader(videos, batch_size=1, shuffle=False, collate_fn= lambda x: x)
    for video in tqdm(loader):
        video_path = video[0]["path"]
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        detections_path = video[0]["path"][20:-3] + "jpg"
        detections = net.detect_on_image(frame, TARGET_SIZE, device, is_pad=False, keep_thresh=CONF_THRESH)
        indices = np.rint(detections.flatten()[:4]).astype(int)
        cropped = frame[indices[1]:indices[3], indices[0]:indices[2], :]
        cv2.imwrite(os.path.join(DETECTIONS_PATH, detections_path), cropped)

if __name__ == "__main__":
    main()