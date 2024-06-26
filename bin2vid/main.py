from __future__ import annotations

from typing import Iterable
import os
from multiprocessing.pool import Pool
import shutil
import base64

import numpy as np
import cv2 as cv


from .encode import (
    split_bytes,
    bytes_to_simplified_matrix,
    matrices_to_real_image,
    matrices_to_real_rgb_image,
    MATRIX_PER_FRAME,
)
from .decode import (
    read_frame,
)
from .video_manipulation import (
    read_video,
    make_video,
)
from .miscs import (
    ceil_div,
    check_file_or_folder_existance,
)


####################
# IMAGE PROCESSING #
####################


def encode_to_image(
    data: bytes,
    images_folder: str = "_tmp",
    max_bytes_per_matrix: int | None = None,
    use_base64: bool = True,
    processes: int | None = 8,
    is_rgb: bool = True
) -> None:
    """processes가 None이라면 encode_to_image가, int라면 encode_to_image_in_parallel가 사용됩니다."""
    if use_base64:
        data = base64.b64encode(data)

    if processes is None:
        _encode_to_image_in_serial(data, images_folder, max_bytes_per_matrix, is_rgb)
    else:
        _encode_to_image_in_parallel(data, images_folder, max_bytes_per_matrix, processes, is_rgb)


def _encode_to_image_in_serial(
    data: bytes,
    images_folder: str = "_tmp",
    max_bytes_per_matrix: int | None = None,
    is_rgb: bool = True,
) -> None:
    check_file_or_folder_existance(images_folder)
    os.makedirs(images_folder)

    zfill_length = len(str(sum(1 for _ in split_bytes(data, max_bytes_per_matrix))))
    image_matrices = (bytes_to_simplified_matrix(bytes_data) for bytes_data in split_bytes(data, max_bytes_per_matrix))
    for i, real_image in enumerate(
        matrices_to_real_rgb_image(image_matrices) if is_rgb else matrices_to_real_image(image_matrices), 1
    ):
        cv.imwrite(f"{images_folder}/{i:0{zfill_length}d}.png", real_image)


# def _encode_to_rgb_image_in_serial(
#     data: bytes,
#     images_folder: str = "_tmp",
#     max_bytes_per_matrix: int | None = None,
# ) -> None:
#     check_file_or_folder_existance(images_folder)
#     os.makedirs(images_folder)

#     zfill_length = len(str(sum(1 for _ in split_bytes(data, max_bytes_per_matrix))))
#     image_matrices = (bytes_to_simplified_matrix(bytes_data) for bytes_data in split_bytes(data, max_bytes_per_matrix))
#     for i, real_image in enumerate(matrices_to_real_rgb_image(image_matrices), 1):
#         cv.imwrite(f"{images_folder}/{i:0{zfill_length}d}.png", real_image)


def _process_batched_bytes(batched_data: Iterable[bytes]) -> np.ndarray:
    pixels = [bytes_to_simplified_matrix(bytes_data) for bytes_data in batched_data]
    return next(matrices_to_real_image(pixels))


def _process_batched_bytes_to_rgb(batched_data: Iterable[bytes]) -> np.ndarray:
    pixels = [bytes_to_simplified_matrix(bytes_data) for bytes_data in batched_data]
    return next(matrices_to_real_rgb_image(pixels))


def _encode_to_image_in_parallel(
    data: bytes,
    images_folder: str = "_tmp",
    max_bytes_per_matrix: int | None = None,
    processes: int = 8,
    is_rgb: bool = False,
) -> None:
    check_file_or_folder_existance(images_folder)
    os.makedirs(images_folder)

    batch_count = MATRIX_PER_FRAME * 3 if is_rgb else MATRIX_PER_FRAME

    splited = list(split_bytes(data, max_bytes_per_matrix))
    batched_bytes_list = [
        splited[i:i + batch_count]
        for i in range(0, ceil_div(len(splited), batch_count) * batch_count, batch_count)
    ]
    zfill_length = len(str(len(splited)))
    with Pool(processes) as pool:
        for i, result in enumerate(pool.imap(
            _process_batched_bytes_to_rgb if is_rgb else _process_batched_bytes,
            batched_bytes_list
        )):
            cv.imwrite(f"{images_folder}/{i:0{zfill_length}d}.png", result)


def decode_from_image(images_folder: str = "_tmp", is_base64_used: bool = True) -> bytes:
    images_data = []
    for filename_include_extension in sorted(os.listdir(images_folder)):
        images_data.append(b"".join(read_frame(cv.imread(f'{images_folder}/{filename_include_extension}'))))
    decoded_data = b"".join(images_data)
    return base64.b64decode(decoded_data, validate=True) if is_base64_used else decoded_data


####################
# VIDEO PROCESSING #
####################


def encode_to_video(
    data: bytes,
    video_path: str = '{images_folder}/output.mp4',
    images_folder: str = "_tmp",
    max_bytes_per_matrix: int | None = None,
    use_base64: bool = True,
    processes: int | None = 8,
    delete_images_folder_after_finished: bool = False,
) -> None:
    encode_to_image(data, images_folder, max_bytes_per_matrix, use_base64, processes)
    video_path = video_path.format(images_folder=images_folder, data_length=len(data))
    make_video(video_path, images_folder)
    if delete_images_folder_after_finished:
        shutil.rmtree(images_folder)


def decode_from_video(
    video_path: str = '{images_folder}/output.mp4',
    is_base64_used: bool = True,
) -> bytes:
    return base64.b64decode(read_video(video_path), validate=True) if is_base64_used else read_video(video_path)
