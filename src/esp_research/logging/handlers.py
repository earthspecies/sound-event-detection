"""This module provides factory functions for creating logging handlers. Handlers are
responsible for _where_ messages are sent (console, files, cloud logging, etc.)."""

import logging

# TODO: is there much point in adding a get_file_handler()?

# TODO: Add Slack handler

# TODO: Add a browser-based handler, maybe using websocket?


# TODO: this is still untested and WIP
def get_gcp_handler(
    project: str,
    level: int = logging.NOTSET,  # defer to logger
    resource: dict | None = None,
    labels: dict | None = None,
) -> logging.Handler:
    """
    Create a Google Cloud Platform logging handler.

    Args:
        project: GCP project ID.
        level: Logging level for this handler.
        resource: GCP resource descriptor (e.g., {"type": "global"}).
        labels: Additional labels to attach to log entries.

    Returns:
        Configured CloudLoggingHandler instance.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> handler = get_gcp_handler(
        ...     level="INFO",
        ...     project="my-gcp-project",
        ...     labels={"experiment": "v1.0"}
        ... )
        >>> logger.addHandler(handler)
        >>> logger.setLevel(logging.INFO)
    """

    from google.cloud import logging as cloud_logging

    client = cloud_logging.Client(project=project)
    handler = cloud_logging.handlers.CloudLoggingHandler(
        client,
        name="esp-research",
        resource=resource,
        labels=labels,
    )
    handler.setLevel(level)

    return handler
