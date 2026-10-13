"""irulan: prepare a dataset of your own writing locally, fine-tune your own
irulan model on kyora, and rewrite text with it."""

__version__ = "0.1.0"

from .client import Client  # noqa: E402
from .errors import (  # noqa: E402
    APIConnectionError, APIError, AuthenticationError, ConflictError, FineTuneFailed, IrulanError,
    NotAvailableYet, NotFoundError, PermissionDeniedError, RateLimitError,
)
from .rewrite import DEFAULT_MODEL, SYSTEM_PROMPT, SYSTEM_VERSION  # noqa: E402

__all__ = [
    "Client", "__version__", "SYSTEM_PROMPT", "SYSTEM_VERSION", "DEFAULT_MODEL",
    "IrulanError", "NotAvailableYet", "APIError", "APIConnectionError", "AuthenticationError",
    "PermissionDeniedError", "NotFoundError", "ConflictError", "RateLimitError", "FineTuneFailed",
]
