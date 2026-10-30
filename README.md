# Video Pose Estimation

A small, reproducible video pose-estimation project using the Ultralytics YOLO pose model. It creates an annotated copy of the input and a second video containing the detected COCO keypoints on a black background.

## Requirements

- Python 3.10 or newer
- A short MP4 video with a person visible (`input.mp4` below). Use your own recording or a video you have permission to use.
- Internet access on the first run to download the model weights. The weights are kept outside this repository by `.gitignore`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pose_estimation.py input.mp4
```

If PowerShell blocks activation, run `python -m pip install -r requirements.txt` using the same Python interpreter without activating the environment.

Output files are written to `outputs/pose_annotated.mp4` and `outputs/pose_points_black.mp4`. To change the confidence threshold or output folder:

```powershell
python pose_estimation.py input.mp4 --confidence 0.35 --output-dir results
```

The first run downloads `yolo26n-pose.pt` through Ultralytics. CPU inference works but may be slow; a CUDA-enabled PyTorch setup can speed it up on a supported NVIDIA GPU.

## What the program does

1. Loads a pretrained pose-estimation model.
2. Reads the video frame by frame to limit memory use.
3. Writes the model's annotated frames to one MP4.
4. Draws detected keypoints as white dots on black frames and saves a second MP4.

## Video recording checklist

Record a short, steady landscape video in a well-lit area. Keep the full person in frame and move slowly through a few clear poses (for example, arms raised, a side turn, and a squat). Avoid private or identifiable bystanders unless they consent. Save the clip as `input.mp4` in this folder, then run the command above. Play both output videos before submitting. Do not commit a large video blindly; the assignment form says to place the video in the repository, or upload a large video to Google Drive and put its share link in this README.

## Submission

The linked form requests your **Name**, **Class Roll No**, and **GitHub repository link**. Deadline shown on the form: **10:00 AM, Saturday, 31 October**. Make the repository public or otherwise accessible to the instructor. If the video is too large for GitHub, upload it to Drive, set access so the instructor can view it, and add the link here:

`Demo video (Google Drive): <paste your view link here>`

Submit the GitHub repository URL in the form. Check your name, roll number, link, and Drive permissions before clicking Submit.

## Credit

This is an independently organized implementation of the same video pose-estimation assignment idea. The friend's public notebook was used as a high-level reference for YOLO pose inference and a black-background keypoint visualization: [ruchidhapola/Pose-Estimation-NLP-Assignment-II](https://github.com/ruchidhapola/Pose-Estimation-NLP-Assignment-II). Review your course's collaboration and attribution rules before submission and make sure you can explain the code yourself.
