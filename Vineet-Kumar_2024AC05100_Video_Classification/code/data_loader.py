import os
import cv2
import numpy as np

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
