"""Domain-specific exceptions for the CLI."""


class SprintSpeedImuError(Exception):
    """Base class for user-facing errors."""


class InputFormatError(SprintSpeedImuError):
    """Raised when input data cannot be parsed or normalized."""


class MissingColumnError(InputFormatError):
    """Raised when required IMU columns are missing."""


class InvalidOptionError(SprintSpeedImuError):
    """Raised when CLI options are inconsistent."""


class ProcessingError(SprintSpeedImuError):
    """Raised when an analysis step cannot complete reliably."""
