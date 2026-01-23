import os
import cv2
import numpy as np

#Function for loading the dataset:

def load_split_data_file(split_file, dataset_path):
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
            parts = line.split(",")        #  CSV split
            rel_path = parts[1].lstrip("/")  # PullUps/v_XXX.avi
            class_name = parts[2]
            

            if class_name not in class_map:
                class_map[class_name] = class_id
                class_id += 1

            video_paths.append(os.path.join(dataset_path, rel_path))
            labels.append(class_map[class_name])

    return video_paths, labels, class_map


# Funtion for extracting video frames: 
# Keeping max number of frames = 30
# keeping frame size = "224 x 224"

def extract_video_frames(video_path, max_frames=30, size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frames_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames_list.append(frame)

    cap.release()

# Using uniform sampling to extract the frames. Max frame: 30

    if len(frames_list) > max_frames:
        idx = np.linspace(0, len(frames_list)-1, max_frames).astype(int)
        frames_list = [frames_list[i] for i in idx]

    return frames_list