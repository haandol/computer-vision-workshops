"""SageMaker Processing script to extract frame images from videos"""

# Built-Ins:
import argparse
import os
import shutil
import subprocess
import sys

# OpenCV requires some OS-level dependencies not in the standard container:
subprocess.call(["apt-get", "update"])
subprocess.call(["apt-get", "-y", "install", "libglib2.0", "libsm6", "libxext6", "libxrender-dev"])
subprocess.call([sys.executable, "-m", "pip", "install", "opencv-python"])
# (or `opencv-contrib-python` if contrib modules required)

# External Dependencies:
import cv2

SUPPORTED_EXTENSIONS = ("avi", "mp4")

def existing_folder_arg(raw_value):
    """argparse type for a folder that must already exist"""
    value = str(raw_value)
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError("%s is not a directory" % value)
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frame images from video files")
    parser.add_argument("--input", type=existing_folder_arg,
        default="/opt/ml/processing/input/videos",
        help="Source folder of video files."
    )
    parser.add_argument("--output", type=str, default="/opt/ml/processing/frames",
        help="Target folder for saving frame images."
    )
    parser.add_argument("--frames-per-second", type=float, default=0,
        help="(Approximate) number of frames per second to save, or save every frame if 0."
    )
    
    return parser.parse_args()

def extract_frames(src, dest, fps=0):
    # Getting the FPS of a video is major_ver dependent in OpenCV:
    (cv_major_ver, _, _) = (cv2.__version__).split('.')
    
    if fps != 0:
        raise NotImplementedError("FPS is a parameter, but not yet implemented!")
    
    for filename in os.listdir(src):
        vidid, _, extension = filename.rpartition(".")
        if (extension not in SUPPORTED_EXTENSIONS):
            print(f"Skipping non-video file {filename}")
            continue
        vidid = filename.rpartition(".")[0]
        vidcap = cv2.VideoCapture(f"{src}/{filename}")#f"data/raw/train_sample_videos/{vidid}.mp4")
        vidfps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS if int(cv_major_ver) < 3 else cv2.CAP_PROP_FPS)
        try:
            shutil.rmtree(f"{dest}/{vidid}")
        except FileNotFoundError:
            pass
        os.makedirs(f"{dest}/{vidid}", exist_ok=True)
        count = 0
        success = True
        success, image = vidcap.read()
        if success:
            print(f"Extracting {filename}", end="")
        else:
            print(f"Error extracting {filename}: Couldn't read capture!", end="")
        while success:
            if (not count % 25):
                print(".", end="")
            cv2.imwrite(f"{dest}/{vidid}/frame-{count:08}.jpg", image)
            success, image = vidcap.read()
            count += 1
        print()
        print(f"Captured {count} frames")
    print(f"Output to {dest}:")
    print(os.listdir(dest))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.makedirs(args.output, exist_ok=True)
    extract_frames(args.input, args.output, fps=args.frames_per_second)
