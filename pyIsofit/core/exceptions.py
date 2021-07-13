""" Custom errors thrown by the package"""

class pgError(Exception):
    """Base error raised by pyIsofit"""

class ParameterError(pgError):
    """Raised when one of the parameters is unsuitable"""