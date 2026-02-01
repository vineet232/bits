import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset

############################################ Classical Machine Learning ###################################################

#Function for loading the dataset:
#The dataset consists the videos from 3 classes:
#Class-1: PullUps, Class-2: Punch, Class-3: PushUps

def load_dataset(split_file, dataset_path):
    video_paths = []
    labels = []
    class_map = {}
    class_id = 0

    with open(split_file, "r") as f:
        next(f) #.... Skipping first row of csv files since it consists of column titles.
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split(",")        #  Spiliting the CSV file.
            rel_path = parts[1].lstrip("/")
            class_name = parts[2]
            

            if class_name not in class_map:
                class_map[class_name] = class_id
                class_id += 1

            video_paths.append(os.path.join(dataset_path, rel_path))
            labels.append(class_map[class_name])

    return video_paths, labels, class_map


# Funtion for extracting video frames: 
# Keeping max number of frames = 30
# keeping frame size = "320 x 240"

def extract_frames(video_path, max_frames=30, size=(320,240)):
    cap = cv2.VideoCapture(video_path)
    frames_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for spatial normalization
        frame = cv2.resize(frame, size)

        # Reduce noise and compression artifacts (Quality enhancement)
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        # Normalize pixel values to [0,1] range (Frame normalization)
        frame = frame.astype(np.float32) / 255.0

        frames_list.append(frame)

    cap.release()

    # Uniform temporal sampling (limit to max_frames)
    if len(frames_list) > max_frames:
        idx = np.linspace(0, len(frames_list)-1, max_frames).astype(int)
        frames_list = [frames_list[i] for i in idx]

    return frames_list



######################################### Deep Learning ####################################################


import os, cv2, random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset2D(Dataset):
    def __init__(self, csv_file, root_dir, class_map,
                 transform=None, num_frames=16, train=True):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.class_map = class_map
        self.transform = transform
        self.num_frames = num_frames
        self.train = train

    def __len__(self):
        return len(self.data)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Quality enhancement (artifact / compression noise)
            frame = cv2.GaussianBlur(frame, (3,3), 0)

            frames.append(frame)

        cap.release()
        return frames


    
    def _sample_frames(self, frames):
        total = len(frames)

        # -------- Dense sampling (every frame) --------
        if total <= self.num_frames:
            idxs = np.linspace(0, total-1, self.num_frames).astype(int)

        else:
            if self.train:
                # 🔹 Random dense clip
                start = random.randint(0, total - self.num_frames)
            else:
                # 🔹 Center dense clip (evaluation)
                start = (total - self.num_frames) // 2

            idxs = np.arange(start, start + self.num_frames)

        return [frames[i] for i in idxs]


    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_rel_path = row["clip_path"]
        label_name = row["label"]

        video_path = os.path.join(self.root_dir, video_rel_path.lstrip("/"))
        label = self.class_map[label_name]

        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)

        processed = []
        for f in frames:
            if self.transform:
                f = self.transform(f)
            processed.append(f)

        video = torch.stack(processed)
        return video, label
    

class VideoDataset3D(VideoDataset2D):
    def __getitem__(self, idx):
        video, label = super().__getitem__(idx)   # (T,C,H,W)
        video = video.permute(1,0,2,3)             # (C,T,H,W) for 3D CNN
        return video, label



