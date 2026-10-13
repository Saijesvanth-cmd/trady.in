class AppError(Exception):
    status = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequest(AppError):
    status, code = 400, "bad_request"


class UnsupportedInterval(BadRequest):
    code = "unsupported_interval"


class ProviderError(AppError):
    status, code = 502, "provider_error"


class ProviderConfigError(ProviderError):
    status, code = 503, "provider_not_configured"


class ProviderAuthError(ProviderError):
    status, code = 503, "provider_auth_failed"


class RateLimited(ProviderError):
    status, code = 429, "provider_rate_limited"


class InstrumentNotFound(ProviderError):
    status, code = 404, "instrument_not_found"


class CandleValidationError(ProviderError):
    code = "invalid_provider_data"
