import os
import shutil


def ceil_div(dividend: int, divisor: int) -> int:
    # div, mod = divmod(dividend, divisor)
    # return div + bool(mod)

    # 더 효율적인 알고리즘. https://stackoverflow.com/a/17511341/21997874
    return -(dividend // -divisor)


def check_file_or_folder_existance(path: str) -> None:
    if os.path.exists(path):
        folder_message = 'folder ' if os.path.isdir(path) else ''
        if input(f"{folder_message}{path!r} already exists. Delete and continue? [y/N]: ") not in {'y', 'Y', 'ㅛ'}:
            raise ValueError(f"User has rejected to continue. Deleting {folder_message}{path!r} might help.")

        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
