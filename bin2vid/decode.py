from __future__ import annotations

import logging
from pylibdmtx import pylibdmtx as matrix
import numpy as np

from .encode import (
    mix_same,
)


def read_frame(frame_image_matrix: np.ndarray, attempt: int = 2) -> list[bytes]:
    data = []
    for y_idx in range(2):
        for x_idx in range(4):
            padded_img = np.pad(frame_image_matrix[480 * y_idx:480 * (y_idx + 1), 480 * x_idx:480 * (x_idx + 1), :],
                                ((10, 10), (10, 10), (0, 0)), constant_values=255)
            decoded = matrix.decode(padded_img, max_count=1)

            if not decoded:
                if attempt <= 0:
                    logging.warning(
                        f"Failed to read frame at {(y_idx, x_idx)=}."
                        "Stop the function and return the information it has received."
                    )
                    return data
                else:
                    logging.warning(f"Retring to read the frame at {(y_idx, x_idx)=}.")

                    # 처음부터 다시 시작.
                    return read_frame(frame_image_matrix, attempt=attempt - 1)

            data.append(decoded[0].data)
    return data


def read_rgb_frame(frame_image_matrix: np.ndarray, attempt: int = 2) -> list[bytes]:
    data = []
    for i in range(3):
        data += read_frame(mix_same(frame_image_matrix[..., i], 3, True, False), attempt)
    return data
