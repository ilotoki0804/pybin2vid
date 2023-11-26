from __future__ import annotations

from .main import (
    encode_to_image,
    check_images_folder,
    decode_from_image,
    encode_to_video,
    decode_from_video,
)


def test_video_coding(
    data: bytes,
    video_path: str = '{images_folder}/output.mp4',
    images_folder: str = "_tmp",
    max_bytes_per_matrix: int | None = None,
    use_base64: bool = True,
    processes: int | None = 8,
) -> bytes:
    encode_to_video(data, video_path, images_folder, max_bytes_per_matrix, use_base64, processes)
    return decode_from_video(video_path.format(images_folder=images_folder), use_base64)


def test_image_coding(
    data: bytes,
    images_folder: str = "_tmp",
    max_bytes_per_matrix: int | None = None,
    use_base64: bool = True,
    processes: int | None = 8,
) -> bytes:
    encode_to_image(data, images_folder, max_bytes_per_matrix, use_base64, processes)
    return decode_from_image(images_folder, use_base64)
