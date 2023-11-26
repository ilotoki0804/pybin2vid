from __future__ import annotations

import shutil

import cv2 as cv
import os

from .decode import (
    read_frame,
)


def make_video(filename_include_extension: str, path_to_images: str) -> None:
    assert shutil.which('ffmpeg') is not None, (
        "ffmpeg is not installed. Install ffmpeg and try again.")

    if os.path.exists(filename_include_extension):
        raise FileExistsError(f"Filename {filename_include_extension} is already taken.")

    os.system(
        f"ffmpeg -framerate 30 -pattern_type glob -i '{path_to_images}'  "
        f"-c:a copy -shortest -c:v libx264 -pix_fmt yuv420p {filename_include_extension}"
    )


def read_video(file_path: str) -> bytes:
    if os.path.isfile(file_path):
        cap = cv.VideoCapture(file_path)
    else:
        raise FileNotFoundError("File not found or it's not a file.")

    images_data = []
    while True:
        retval, frame = cap.read()
        if not retval:
            break

        images_data.append(b"".join(read_frame(frame)))
        print(".", end='')

    if cap.isOpened():
        cap.release()

    decoded_data = b"".join(images_data)
    return decoded_data
