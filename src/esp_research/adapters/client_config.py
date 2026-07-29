"""Configuration for the HTTP adapter client."""

from pydantic import BaseModel, Field

from esp_research.configs.cli_config import CLIConfig


class HttpAuthConfig(BaseModel):
    """Authentication configuration for HTTP requests.

    Attributes
    ----------
    header : str
        The HTTP header name for authentication (e.g. "Authorization").
    value : str
        The header value. Supports environment variable references
        like "Bearer ${OPENAI_API_KEY}".
    """

    header: str = "Authorization"
    value: str


class HttpClientConfig(CLIConfig):
    """Configuration for the HTTP adapter client.

    Attributes
    ----------
    url : str
        Base URL of the adapter service, including the adapter-type prefix
        (e.g. ``"http://localhost:8001/openai"``).
    route : str | None
        Specific route to POST inference requests to (e.g.
        ``"beans-zero-forgiving"``).  The full request URL is
        ``{url}/{route}``.  When ``None``, requests are sent directly to
        ``url``.
    auth : HttpAuthConfig | None
        Optional authentication configuration.
    timeout : float
        Request timeout in seconds.
    retries : int
        Number of retry attempts on transient failures (5xx, 429).
    audio_key : str | None
        Key in ``input_data`` whose value holds raw audio bytes for
        base64 encoding before sending the request.  When ``None``,
        no audio encoding is performed and the payload is sent as-is.
    audio_format : str | None
        Audio format label (e.g. ``"wav"``, ``"mp3"``).  Included in
        the request payload alongside the encoded audio when set.
        When ``None``, no ``audio_format`` key is added to the payload.
    """

    url: str
    route: str | None = None
    auth: HttpAuthConfig | None = None
    timeout: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0)
    audio_key: str | None = None
    audio_format: str | None = None
