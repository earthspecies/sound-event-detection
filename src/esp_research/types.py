"""Shared type aliases for the esp-research package."""

from alp_data.io import AnyPathT

# A filesystem path in any accepted form: a concrete path object (a local
# `pathlib.Path` or a cloud path from `alp_data.io`) or a plain string.
# `AnyPathT` already includes `Path`, so it is not listed separately.
type AnyPathOrStr = AnyPathT | str

__all__ = ["AnyPathOrStr"]
