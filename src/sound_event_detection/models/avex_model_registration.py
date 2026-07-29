"""Register non-public avex models at import time.

Reads model configs from avex's internal package resources (avex/api/configs/checkpoints/)
and registers them under friendly names in the avex model registry, so they can be loaded
by name via load_model() (e.g. load_model("beats_ssl", return_features_only=True)).

The checkpoints/ configs are not registered as official user-facing models in avex, but
their ModelSpec is valid and the BEATs backbone resolves checkpoint weights internally via
_get_beats_checkpoint_path(), so no explicit checkpoint_path is needed here.

frame_detector.py imports this module at module level, ensuring registration happens
before any load_model("beats_ssl") call in the training/inference pipeline.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import logging

import avex
import yaml
from avex.configs import ModelSpec

logger = logging.getLogger(__name__)

# Checkpoint paths for models registered here that are not in avex's official_models/.
# Populated during registration; consumed by create_beats_detector to pass checkpoint_path
# explicitly to load_model (since get_checkpoint_path() only reads official_models/).
#
# IMPORTANT: avex's load_model() silently builds a randomly-initialised backbone when no
# checkpoint is found — it does NOT raise.  Any model registered below MUST have a
# checkpoint path stored here, and create_beats_detector() will refuse to proceed if the
# entry is missing, preventing silent training on random weights.
_REGISTERED_CHECKPOINT_PATHS: dict[str, str] = {}

# Models that must have a checkpoint path; training is aborted if the path is missing.
_CHECKPOINT_REQUIRED: set[str] = {"beats_ssl"}

_BEATS_CHECKPOINT_CONFIGS_PKG = "avex.api.configs.checkpoints"
_BEATS_SSL_CONFIG_STEM = "beats_iter3_plus_as2m_ssl"


def _read_beats_checkpoint_yaml(config_stem: str) -> dict | None:
    """Read a beats checkpoint YAML from avex/api/configs/checkpoints/.

    Returns:
        Parsed YAML data dict, or None if the file could not be read.
    """
    try:
        from avex.models.utils.registry import load_packaged_yaml_mapping

        return load_packaged_yaml_mapping(package=_BEATS_CHECKPOINT_CONFIGS_PKG, name=config_stem)
    except Exception:
        logger.debug("Could not read beats checkpoint YAML: %s", config_stem, exc_info=True)
        return None


def _register_from_avex_checkpoint_config(model_name: str, config_stem: str) -> bool:
    """Register a model by reading an avex-internal checkpoint YAML.

    Handles two YAML formats:
    1. Standard: has a ``model_spec`` key with a complete ModelSpec dict.
    2. BEATs checkpoint: has ``beats_cfg`` + optional ``checkpoint_path``; builds a
       ModelSpec from the base ``esp_aves2_sl_beats_all`` spec and the SSL beats_cfg.

    Args:
        model_name: Name to register the model under in the avex registry.
        config_stem: Filename stem (without .yml) inside avex/api/configs/checkpoints/.

    Returns:
        True if registration succeeded, False otherwise.
    """
    try:
        avex_root = pkg_resources.files("avex")
        config_file = (
            avex_root.joinpath("api").joinpath("configs").joinpath("checkpoints").joinpath(f"{config_stem}.yml")
        )
        with config_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "model_spec" in data:
            spec = ModelSpec(**data["model_spec"])
        elif "beats_cfg" in data:
            # BEATs checkpoint YAML — build a proper spec using the SSL beats_cfg so the
            # model is constructed with the correct SSL architecture (not the finetuned one).
            base_spec = avex.get_model_spec("esp_aves2_sl_beats_all")
            spec = base_spec.model_copy(
                update={
                    "pretrained": False,
                    "fine_tuned": False,
                    "init_config": data["beats_cfg"],
                }
            )
            if isinstance(data.get("checkpoint_path"), str) and data["checkpoint_path"]:
                _REGISTERED_CHECKPOINT_PATHS[model_name] = data["checkpoint_path"]
                logger.info("Stored checkpoint path for '%s': %s", model_name, data["checkpoint_path"])
        else:
            return False

        avex.register_model(model_name, spec)
        logger.info("Registered '%s' from avex internal config '%s.yml'", model_name, config_stem)
        return True
    except Exception:
        logger.debug(
            "Could not register '%s' from checkpoints/%s.yml (avex may not have this file in this version)",
            model_name,
            config_stem,
            exc_info=True,
        )
        return False


def _register_beats_ssl_fallback() -> bool:
    """Fallback: derive beats_ssl spec from esp_aves2_sl_beats_all via model_copy.

    Used when avex/api/configs/checkpoints/ is not present (e.g. avex ≤ 1.1.0).
    Reads the SSL beats_cfg and checkpoint_path directly from the checkpoints YAML so
    the model is built with the correct SSL architecture and weights.

    Returns:
        True if registration succeeded, False otherwise.
    """
    try:
        base_spec = avex.get_model_spec("esp_aves2_sl_beats_all")

        # Try to read the SSL config and checkpoint path from the packaged YAML.
        ssl_yaml = _read_beats_checkpoint_yaml(_BEATS_SSL_CONFIG_STEM)
        ssl_init_config = ssl_yaml.get("beats_cfg") if ssl_yaml else None
        ssl_checkpoint_path = ssl_yaml.get("checkpoint_path") if ssl_yaml else None

        update: dict = {"pretrained": False, "fine_tuned": False}
        if ssl_init_config is not None:
            update["init_config"] = ssl_init_config

        ssl_spec = base_spec.model_copy(update=update)
        avex.register_model("beats_ssl", ssl_spec)

        if isinstance(ssl_checkpoint_path, str) and ssl_checkpoint_path:
            _REGISTERED_CHECKPOINT_PATHS["beats_ssl"] = ssl_checkpoint_path
            logger.info("Stored SSL checkpoint path for 'beats_ssl': %s", ssl_checkpoint_path)

        logger.info("Registered 'beats_ssl' via model_copy fallback from 'esp_aves2_sl_beats_all'")
        return True
    except Exception:
        logger.error("Fallback registration of 'beats_ssl' also failed", exc_info=True)
        return False


def register_beats_ssl() -> None:
    """Register 'beats_ssl' in the avex model registry.

    Tries the checkpoints/ YAML first; falls back to model_copy if unavailable.

    Raises:
        RuntimeError: If all registration strategies fail, or if a checkpoint path
            could not be resolved for a model that requires one.
    """
    if _register_from_avex_checkpoint_config("beats_ssl", _BEATS_SSL_CONFIG_STEM):
        pass
    elif _register_beats_ssl_fallback():
        pass
    else:
        raise RuntimeError(
            "Could not register 'beats_ssl' in the avex model registry. Check logs above for the root cause."
        )

    # Hard fail at import time rather than silently training on random weights.
    if "beats_ssl" not in _REGISTERED_CHECKPOINT_PATHS:
        raise RuntimeError(
            "beats_ssl was registered in the avex model registry but no SSL checkpoint path "
            "could be resolved.  avex.load_model('beats_ssl') would silently build a "
            "randomly-initialised backbone and training would produce garbage results.  "
            "Ensure avex >= version that ships beats_iter3_plus_as2m_ssl.yml in "
            "avex/api/configs/checkpoints/, or set the checkpoint path manually."
        )


register_beats_ssl()
