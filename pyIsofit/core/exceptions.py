""" Custom errors thrown by the package"""


class pgError(Exception):
    """Base error raised by pyIsofit"""


class ParameterError(pgError):
    """Raised when one of the parameters is unsuitable"""


class SaveError(pgError):
    """Raised when the .save method is called incorrectly"""
