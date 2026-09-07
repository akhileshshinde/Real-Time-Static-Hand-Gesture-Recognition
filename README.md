# Real-Time Static Hand Gesture Recognition

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Model](https://img.shields.io/badge/model-YOLOv5-orange)

Real-time detection of custom static hand gestures — open palm, fist, peace sign,
thumbs up — from a live webcam, RTSP stream, or recorded video, using a YOLOv5 model
trained on a manually collected and annotated dataset.

## Why YOLO

- Fast enough for real-time, frame-by-frame inference
- High accuracy even on a small, custom-collected dataset
- Straightforward transfer learning from pretrained weights
- Mature tooling and community support

## Dataset & preprocessing

All gesture data was collected manually via webcam and hand-annotated with LabelImg.
Frames are preprocessed before training/inference to make the hand shape more
distinguishable against the background:

```
raw frame ──► grayscale ──► binary threshold ──► model input
```

## Training details

| | |
|---|---|
| Framework | YOLOv5 |
| Image size | 640x640 |
| Train/val split | 80/20 |
| Optimizer | SGD |
| Loss | YOLOv5 default (CIoU + BCE) |
| Classes | `open_palm`, `fist`, `peace_sign`, `thumbs_up` |

## Setup

```bash
git clone https://github.com/akhileshshinde/Real-Time-Static-Hand-Gesture-Recognition
cd Real-Time-Static-Hand-Gesture-Recognition
pip install -r requirements.txt
```

## Usage

```bash
python video_test_hand_sign.py
```

Update the model path in `video_test_hand_sign.py` to point at your trained weights
(`best.pt`). The script preprocesses each frame with the same thresholding used in
training, runs YOLOv5 inference, and overlays detections on the original RGB feed.

## Demo

[`output_video.mkv`](output_video.mkv) shows live detection results.

## Future improvements

- Train directly on raw RGB frames to remove the preprocessing/thresholding dependency
- Add data augmentation (noise, rotation, brightness) for robustness
- Expand the gesture vocabulary (two-hand gestures, rotations)
- Optimize for edge deployment (Jetson Nano, mobile)

## License

MIT — see [LICENSE](LICENSE).
