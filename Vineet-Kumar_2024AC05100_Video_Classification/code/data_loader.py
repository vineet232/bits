import os
import cv2
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset


############################################
#      Classical Machine Learning Utils
############################################

# This function reads a CSV split file and prepares
# absolute video paths along with numeric class labels.
# The dataset contains three action classes:
# PullUps, Punch, PushUps

def load_dataset(split_file, dataset_path):
    video_paths = []     # stores full paths to video files
    labels = []          # stores integer-encoded labels
    class_map = {}       # maps class name → class index
    class_id = 0         # incremental class counter

    with open(split_file, "r") as f:
        # Skip header row (column names)
        next(f)

        for line in f:
            line = line.strip()
            if line == "":
                continue

            # Parse CSV row
            parts = line.split(",")
            rel_path = parts[1].lstrip("/")
            class_name = parts[2]

            # Assign a numeric ID to each new class
            if class_name not in class_map:
                class_map[class_name] = class_id
                class_id += 1

            # Build absolute video path
            video_paths.append(os.path.join(dataset_path, rel_path))
            labels.append(class_map[class_name])

    return video_paths, labels, class_map


# This function extracts frames from a video file.
# - Caps the number of frames for temporal consistency
# - Resizes frames for spatial uniformity
# - Normalizes pixel values for ML pipelines

def extract_frames(video_path, max_frames=30, size=(320,240)):
    cap = cv2.VideoCapture(video_path)
    frames_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to fixed resolution
        frame = cv2.resize(frame, size)

        # Apply slight blur to suppress noise/artifacts
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        # Scale pixel values to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        frames_list.append(frame)

    cap.release()

    # Temporal downsampling if video is longer than max_frames
    if len(frames_list) > max_frames:
        idx = np.linspace(0, len(frames_list) - 1, max_frames).astype(int)
        frames_list = [frames_list[i] for i in idx]

    return frames_list


############################################
#           Deep Learning Datasets
############################################

class VideoDataset2D(Dataset):
    # Dataset class for 2D CNN.
    # Frames are sampled from videos and processed
    # independently before temporal stacking.

    def __init__(self, csv_file, root_dir, class_map,
                 transform=None, num_frames=16, train=True):

        # Load metadata CSV
        self.data = pd.read_csv(csv_file)

        # Root directory containing video files
        self.root_dir = root_dir

        # Mapping from class name to label index
        self.class_map = class_map

        # Optional image transformations
        self.transform = transform

        # Number of frames per video clip
        self.num_frames = num_frames

        # Flag to control train vs evaluation sampling
        self.train = train

    def __len__(self):
        # Total number of video samples
        return len(self.data)

    def _load_video(self, path):
        
        # Reads all frames from a video file.
        # No temporal sampling is done here.
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV BGR format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Light blur to reduce compression noise
            frame = cv2.GaussianBlur(frame, (3,3), 0)

            frames.append(frame)

        cap.release()
        return frames


    def _sample_frames(self, frames):
        
        # Selects a fixed-length clip from the full video.
        # Uses random sampling during training and
        # center sampling during evaluation.
        
        total = len(frames)

        # If video is shorter, interpolate indices
        if total <= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames).astype(int)
        else:
            if self.train:
                # Random contiguous clip for training
                start = random.randint(0, total - self.num_frames)
            else:
                # Center clip for validation/testing
                start = (total - self.num_frames) // 2

            idxs = np.arange(start, start + self.num_frames)

        return [frames[i] for i in idxs]


    def __getitem__(self, idx):
        
        # Loads a video sample and its label.
        # Returns frames stacked as a tensor.
        
        row = self.data.iloc[idx]

        video_rel_path = row["clip_path"]
        label_name = row["label"]

        # Construct full video path
        video_path = os.path.join(self.root_dir, video_rel_path.lstrip("/"))

        # Convert class name to numeric label
        label = self.class_map[label_name]

        # Load and sample frames
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)

        processed = []
        for f in frames:
            if self.transform:
                f = self.transform(f)
            processed.append(f)

        # Shape: (T, C, H, W)
        video = torch.stack(processed)

        return video, label


class VideoDataset3D(VideoDataset2D):
    
    # Extension of VideoDataset2D for 3D CNNs.
    # Rearranges tensor dimensions to match
    # (C, T, H, W) format expected by 3D models.
    

    def __getitem__(self, idx):
        video, label = super().__getitem__(idx)   # (T, C, H, W)

        # Reorder dimensions for 3D convolution
        video = video.permute(1, 0, 2, 3)         # (C, T, H, W)

        return video, label
