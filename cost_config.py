"""
Cost-assumption loading.

Broker friction (commissions, spread, slippage) is read from a TOML file
rather than hardcoded, so account-specific numbers stay out of version
control. `config/costs.toml` is gitignored; only
`config/costs.example.toml` (placeholders) is committed.

Resolution order:
  1. $MES_COSTS_CONFIG, if set
  2. config/costs.toml
  3. config/costs.example.toml  -- placeholders only, warns on use
"""

from __future__ import annotations

import os
import tomllib
import warnings
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
REAL_COSTS = _CONFIG_DIR / "costs.toml"
EXAMPLE_COSTS = _CONFIG_DIR / "costs.example.toml"


class CostConfigError(RuntimeError):
    """Raised when no cost configuration can be loaded."""


def load_costs() -> dict:
    override = os.environ.get("MES_COSTS_CONFIG")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise CostConfigError(f"MES_COSTS_CONFIG points at {path}, which does not exist.")
        is_placeholder = False
    elif REAL_COSTS.is_file():
        path, is_placeholder = REAL_COSTS, False
    elif EXAMPLE_COSTS.is_file():
        path, is_placeholder = EXAMPLE_COSTS, True
    else:
        raise CostConfigError(
            f"No cost configuration found. Copy {EXAMPLE_COSTS.name} to "
            f"{REAL_COSTS.name} in {_CONFIG_DIR}."
        )

    with path.open("rb") as fh:
        costs = tomllib.load(fh)

    if is_placeholder:
        warnings.warn(
            f"Loaded PLACEHOLDER cost assumptions from {path.name} -- all "
            "friction is zero, which will overstate every result. Copy it to "
            "config/costs.toml and set your broker's real numbers.",
            RuntimeWarning,
            stacklevel=2,
        )

    costs["_source"] = str(path)
    costs["_is_placeholder"] = is_placeholder
    return costs


COSTS = load_costs()
