class Bin2VidError(Exception):
    """Parent class of all bin2vid exceptions.

    Note that not *every* error in bin2vid is not child class of this exception.
    Maybe AssertionError, ValueError, or any other can be invoked.
    """
