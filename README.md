# Video Pose Estimation

A reproducible video pose-estimation project using the Ultralytics YOLO pose model. It creates an annotated video and a second video with detected COCO keypoints on a black background.

## Demo videos

- [Pose estimation with annotations](pose_annotated.mp4)
- [Detected keypoints on black](pose_points_black.mp4)
- Input clip: [input.mp4](input.mp4)

## Requirements

- Python 3.10 or newer
- Internet access on the first run to download the model weights. The weights are kept outside this repository by `.gitignore`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pose_estimation.py input.mp4
```

Output files are written to `outputs/pose_annotated.mp4` and `outputs/pose_points_black.mp4`. To change the confidence threshold or output folder:

```powershell
.\.venv\Scripts\python.exe pose_estimation.py input.mp4 --confidence 0.35 --output-dir results
```

The first run downloads `yolo26n-pose.pt` through Ultralytics. CPU inference works but may be slow; a CUDA-enabled PyTorch setup can speed it up on a supported NVIDIA GPU.

## What the program does

1. Loads a pretrained pose-estimation model.
2. Reads the video frame by frame to limit memory use.
3. Writes annotated frames to one MP4.
4. Draws detected keypoints as white dots on black frames and saves a second MP4.

## Recording checklist

For another demo clip, record a short, steady landscape video in a well-lit area. Keep the full person in frame and move through a few clear poses. Use a video you have permission to share, and avoid bystanders unless they consent. Play the generated videos before submission.

The assignment form says the repository should contain the code and video. If a future video is too large for GitHub, upload it to Google Drive, allow the instructor to view it, and put its link here.

## Credit

This is an independently organized implementation of the same video pose-estimation assignment idea. The friend's public notebook was used as a high-level reference for YOLO pose inference and a black-background keypoint visualization: [ruchidhapola/Pose-Estimation-NLP-Assignment-II](https://github.com/ruchidhapola/Pose-Estimation-NLP-Assignment-II). Review your course's collaboration and attribution rules and make sure you can explain the code yourself.
