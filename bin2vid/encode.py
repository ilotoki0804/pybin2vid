from __future__ import annotations
import itertools

from typing import Iterable, Iterator, Sequence, TypeVar
import string

from pylibdmtx import pylibdmtx as matrix
import numpy as np

from .miscs import (
    ceil_div,
    check_file_or_folder_existance,
)

T = TypeVar('T')

MATRIX_PER_FRAME = 8
DEFAULT_MAX_BYTES_PER_MATRIX = 660


def split_bytes(bytes_data: bytes, max_bytes_per_matrix: int | None = DEFAULT_MAX_BYTES_PER_MATRIX) -> Iterator[bytes]:
    assert set(bytes_data).issubset(string.printable.encode('ascii')), (
        "Not all of data is printable. In order to encode bytes to matrix, "
        f"all of bytes should be printable.\nPrintable bytes: {string.printable}\n"
        "Use base64(stdlib) to convert them to printable bytes."
    )

    max_bytes_per_matrix = max_bytes_per_matrix or DEFAULT_MAX_BYTES_PER_MATRIX

    total_matrix_count = ceil_div(
        ceil_div(len(bytes_data), max_bytes_per_matrix), MATRIX_PER_FRAME) * MATRIX_PER_FRAME
    data_per_matrix = ceil_div(len(bytes_data), total_matrix_count)

    assert total_matrix_count * data_per_matrix >= len(bytes_data), (
        f"{total_matrix_count=} * {data_per_matrix=} < {len(bytes_data)=}")

    for i in range(0, len(bytes_data), data_per_matrix):
        yield bytes_data[i:i + data_per_matrix]


def bytes_to_simplified_matrix(bytes_data: bytes) -> np.ndarray:
    try:
        encoded = matrix.encode(bytes_data, size="120x120")
    except matrix.PyLibDMTXError as e:
        raise ValueError("This error is occured when data length is too long. "
                         "Decrease max_bytes_per_matrix may be solve problem. "
                         f"Data length: {len(bytes_data)}") from e

    # 정보를 numpy로 넘기고 반복되는 정보를 제거해 단순화함.
    pixels = np.fromiter(encoded[-1], np.int16).reshape((-1, 3))[:, 1]
    return pixels.reshape((620, 620))[::5, ::5][2:-2, 2:-2]


def batch_and_fill_rest(
    simplified_matrices: Iterable[T],
    batch_size: int = MATRIX_PER_FRAME,
    empty_alternative: T | None = None,
) -> Iterator[list[T]]:
    simplified_matrices = iter(simplified_matrices)

    is_empty_value_initialized = False
    continue_progress = True
    while continue_progress:
        batched_matrices: list[T] = []
        for _ in range(batch_size):
            try:
                batched_matrices.append(next(simplified_matrices))
            except StopIteration:
                if not batched_matrices:  # 크기가 정확하게 딱 맞았을 때
                    return
                assert empty_alternative is not None, "Maybe `simplified_matrices` was empty."
                batched_matrices.append(empty_alternative)
                continue_progress = False

            if is_empty_value_initialized or empty_alternative is not None:
                continue
            sample_matix = batched_matrices[0]
            assert isinstance(sample_matix, np.ndarray), (
                "`simplified_matrices` should be `np.ndarray` type when `empty_value` is None.")
            empty_alternative = np.full(sample_matix.shape, empty_alternative)
            is_empty_value_initialized = True

        yield batched_matrices


def matrices_to_real_image(simplified_matrices: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    for batched_matrices in batch_and_fill_rest(simplified_matrices):
        yield make_real_image(concat_by_grid(batched_matrices, (4, 2)))


def batch_rgb(simplified_matrices: Iterable[np.ndarray]) -> Iterator[list[np.ndarray]]:
    simplified_matrices = iter(simplified_matrices)
    empty_alternative = [np.full((120, 120), 225)] * MATRIX_PER_FRAME

    for batched_matrices_rgb in batch_and_fill_rest(batch_and_fill_rest(simplified_matrices), 3, empty_alternative):
        concatenated_matrices_rgb = [concat_by_grid(batched_matrices, (4, 2))
                                     for batched_matrices in batched_matrices_rgb]
        yield concatenated_matrices_rgb


def matrices_to_real_rgb_image(simplified_matrices: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    for concatenated_matrices_rgb in batch_rgb(simplified_matrices):
        yield make_real_rgb_image(concatenated_matrices_rgb)


def make_real_rgb_image(simplified_matirces: list[np.ndarray]) -> np.ndarray:
    return mix((mix_same(mix_same(simplified_matrix, 4, True), 4, False)
                for simplified_matrix in simplified_matirces), True, False)


def concat_by_grid(arrays: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """
    concatenate order:
    1 2 3 4
    5 6 7 8
    """
    y_shape, x_shape = shape
    if y_shape == -1:
        if len(arrays) % x_shape != 0:
            raise ValueError(f"shape not matched. {len(arrays) = }, {shape = }, array shape = {arrays[0].shape}")
        y_shape = len(arrays) // x_shape
    elif x_shape == -1:
        if len(arrays) % y_shape != 0:
            raise ValueError(f"shape not matched. {len(arrays) = }, {shape = }, array shape = {arrays[0].shape}")
        x_shape = len(arrays) // y_shape

    assert len(arrays) == y_shape * x_shape, (
        f"{len(arrays)} != {y_shape} * {x_shape}")

    to_contatenate = []
    for i in range(0, len(arrays), y_shape):
        to_contatenate.append(np.concatenate(arrays[i:i + y_shape], 1))

    return np.concatenate(to_contatenate, 0)


def mix(arrays: Iterable[np.ndarray], is_x_axis: bool, reshape: bool = True):
    """Mixes two array.

    array1:
        [[1, 2, 3],
         [4, 5, 6]]

    array2:
        [[ 7,  8,  9],
         [10, 11, 12]]

    mix([array1, array2], False, False):
        array([[[ 1,  2,  3],
                [ 7,  8,  9]],
               [[ 4,  5,  6],
                [10, 11, 12]]])

    mix([array1, array2], False, True):
        array([[ 1,  2,  3],
               [ 7,  8,  9],
               [ 4,  5,  6],
               [10, 11, 12]])

    mix([array1, array2], True, False):
        array([[[[ 1,  2,  3]],
                [[ 4,  5,  6]]],
               [[[ 7,  8,  9]],
                [[10, 11, 12]]]])

    mix([array1, array2], True, True):
        array([[[ 1,  2,  3],
                [ 4,  5,  6]],
               [[ 7,  8,  9],
                [10, 11, 12]]])
    """

    # Stack Overflow #33769094
    if is_x_axis:
        mixed_array = np.array([list(zip(*x_cols)) for x_cols in zip(*arrays)])
        return mixed_array.reshape(*mixed_array.shape[:-2], -1) if reshape else mixed_array
    else:
        mixed_array = np.array(list(zip(*arrays)))
        return mixed_array.reshape(-1, *mixed_array.shape[2:]) if reshape else mixed_array


def mix_same(array: np.ndarray, count: int, is_x_axis: bool, reshape: bool = True):
    return mix([array] * count, is_x_axis, reshape)


def make_real_image(simplified_matirx: np.ndarray) -> np.ndarray:
    return mix_same(mix_same(mix_same(simplified_matirx, 4, True), 4, False), 3, True, False)


def add_padding_by_certaion_size(image_matrix: np.ndarray, shape: tuple[int, int], fill_with: int = 256):
    # strict zip이면 안 됨.
    assert all(pixel_dim <= shape_dim for pixel_dim, shape_dim in zip(image_matrix.shape, shape)), (
        f"Shape of pixcels{image_matrix.shape} is bigger than given shape{shape}.")

    # pixels = np.concatenate([np.full((pixels[0], shape[1] - pixels[1], *pixels.shape[2:]), 256)])

    y_shape, x_shape = shape
    y_shape_diff, x_shape_diff = y_shape - image_matrix.shape[0], x_shape - image_matrix.shape[1]

    return np.pad(
        image_matrix,
        ((0, y_shape_diff), (0, x_shape_diff)),
        mode='constant',
        constant_values=np.full(image_matrix.shape[2:], fill_with)
    )
