"""Errors raised by the irulan client."""


class IrulanError(Exception):
    """Base class for every error the client raises."""


class NotAvailableYet(IrulanError):
    """The hosted API (or this part of it) is not live yet.

    Raised when no base URL is configured, or when the server answers a
    planned endpoint with 501 Not Implemented. Point IRULAN_BASE_URL at a
    server that implements docs/API.md to use it today."""


class APIConnectionError(IrulanError):
    """The server could not be reached."""


class APIError(IrulanError):
    """The server answered with an error, in the OpenAI-style shape
    {"error": {"message", "type", "code", "param"}}."""

    def __init__(self, message, status=None, type=None, code=None, param=None, body=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.code = code
        self.param = param
        self.body = body

    def __str__(self):
        bits = [str(self.status)] if self.status else []
        if self.code:
            bits.append(self.code)
        prefix = " ".join(bits)
        return "%s: %s" % (prefix, self.message) if prefix else self.message


class AuthenticationError(APIError):
    """401: missing or invalid API key."""


class PermissionDeniedError(APIError):
    """403: the key may not do this."""


class NotFoundError(APIError):
    """404: no such dataset, fine-tune or model."""


class ConflictError(APIError):
    """409: the object is in the wrong state (e.g. the dataset is not ready)."""


class RateLimitError(APIError):
    """429: slow down."""


class FineTuneFailed(IrulanError):
    """wait() saw the fine-tune end as failed or cancelled."""

    def __init__(self, job):
        self.job = job
        err = (job.get("error") or {}).get("message") if isinstance(job.get("error"), dict) else job.get("error")
        super().__init__("fine-tune %s %s%s" % (job.get("id"), job.get("status"), (": %s" % err) if err else ""))
