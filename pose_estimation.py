import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def process_video(source: Path, annotated_path: Path, skeleton_path: Path, model_name: str, confidence: float) -> None:
    model = YOLO(model_name)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = cv2.VideoWriter(str(annotated_path), codec, fps, (width, height))
    skeleton = cv2.VideoWriter(str(skeleton_path), codec, fps, (width, height))
    if not annotated.isOpened() or not skeleton.isOpened():
        capture.release()
        raise RuntimeError("Could not create output videos. Check the output path and codec support.")

    try:
        for result in model.predict(source=str(source), conf=confidence, stream=True, verbose=False):
            annotated.write(result.plot())
            black = np.zeros((height, width, 3), dtype=np.uint8)
            if result.keypoints is not None:
                points = result.keypoints.xy.cpu().numpy()
                for person in points:
                    for x, y in person:
                        if x > 0 and y > 0:
                            cv2.circle(black, (int(x), int(y)), 4, (255, 255, 255), -1)
            skeleton.write(black)
    finally:
        capture.release()
        annotated.release()
        skeleton.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate human poses in a video with YOLO pose.")
    parser.add_argument("video", type=Path, help="Path to an input video, for example input.mp4")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose model (default: yolo26n-pose.pt)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence from 0 to 1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated videos")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Input video does not exist: {args.video}")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    process_video(args.video, args.output_dir / "pose_annotated.mp4", args.output_dir / "pose_points_black.mp4", args.model, args.confidence)
    print(f"Annotated video: {args.output_dir / 'pose_annotated.mp4'}")
    print(f"Keypoint video:  {args.output_dir / 'pose_points_black.mp4'}")


if __name__ == "__main__":
    main()
