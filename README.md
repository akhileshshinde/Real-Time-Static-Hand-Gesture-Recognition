# Real-Time-Static-Hand-Gesture-Recognition

## ✋ Hand Gesture Recognition with YOLOv5
### 📌 Project Overview

This project implements a real-time hand gesture recognition system using YOLOv5.
The goal is to detect custom hand signs such as open palm, fist, peace sign (V), thumbs up in live video streams (webcam, RTSP, or recorded videos).

A demonstration video is attached in the repo to showcase live detection results.

## 🎯 Why YOLO?

I chose YOLO (You Only Look Once) because:

It is fast and optimized for real-time applications.

It offers high accuracy even on custom small datasets.

Supports easy transfer learning, so I could train on my manually collected dataset.

Well-documented and widely used, making it easier to build upon.

### 🖼️ Dataset Collection & Preprocessing

I manually collected custom gesture data using my webcam.

Each frame was preprocessed with background subtraction and thresholding.

Converted frames to grayscale.

Applied binary thresholding to highlight the hand region.

This reduced background noise and made the hand shape more distinguishable.

#### Example preprocessing:

Original Frame	Thresholded Frame

	

All preprocessed frames were manually annotated using LabelImg
.

#### ⚙️ Training Details

Framework: YOLOv5

Image size: 640x640

Train/Validation split: 80/20

Optimizer: SGD

Loss function: YOLOv5 default (CIoU + BCE Loss)

Hardware: Trained on GPU

### ▶️ Inference Script

The script supports live webcam, RTSP, or video file input.
``` bash
python video_test_hand_sign.py
```
``` bash
model = DetectMultiBackend(
    "(change the model path here)/best.pt",
    device=device,
    dnn=False
)
``` 

Preprocesses input frames with thresholding (to match training data).

Runs YOLOv5 inference on the processed frame.

Displays detections on the original RGB feed.

### 📹 Demo Video - I have attached video in repo.

### 🔮 Improvements & Next Steps

#### Remove preprocessing dependency:
Train the model directly on raw RGB images so background subtraction isn’t required.

#### Data augmentation:
Add random noise, rotation, brightness changes to make the model more robust.

#### Expand gesture vocabulary:
Include more signs and variations (two-hand gestures, rotations).

#### Edge deployment:
Optimize for devices like NVIDIA Jetson Nano or mobile phones.


