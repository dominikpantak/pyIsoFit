""" Custom errors thrown by the package"""


class pgError(Exception):
    """Base error raised by pyIsofit-master"""


class ParameterError(pgError):
    """Raised when one of the parameters is unsuitable"""


class SaveError(pgError):
    """Raised when the .save method is called incorrectly"""


class HenryError(pgError):
    """Raised when there is a problem with the henry regime approximation"""
