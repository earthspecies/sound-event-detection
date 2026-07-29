"""Public-release variant of ``esp_research/__init__.py``.

``scripts/publish_public.sh`` copies this file over the shipped
``src/esp_research/__init__.py``. In this repo ``esp_research`` is not its own
distribution: it ships inside the flat ``sound-event-detection`` distribution.
``__version__`` therefore reads that distribution's metadata (populated from
``pyproject.toml``) instead of a standalone ``esp-research`` one. It is consumed
by ``esp_research.checkpointing`` to stamp provenance into saved checkpoints.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("sound-event-detection")
except PackageNotFoundError:
    __version__ = "0.0.0"
