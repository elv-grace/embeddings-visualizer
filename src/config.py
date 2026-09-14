"""The service's configuration file: `config.yml` beside the repo.

What belongs here
-----------------
Anything a deployment has to set that is not a secret and not a per-request
choice: which container image each model is queried through, and how containers
are run. Image names in particular differ per registry and per retag, and they
are exactly what should not require a patch to `src/`.

YAML rather than JSON so the file can explain itself. That matters for this one:
a wrong image silently returns meaningless neighbours rather than failing, so
the file has to be able to say what each entry is and what depends on it. (YAML
is a superset of JSON, so a JSON file parses here too.)

Precedence
----------
Environment first, then this file, then the built-in default. An env var is the
right tool for a one-off -- pointing a single run at a different runtime or
raising a timeout to see whether a model is merely slow -- and it should not have
to be undone in a file that is shared and committed.

Missing or broken is not fatal: an index still loads, projects and plots with no
configuration at all. Only a *query* needs an image, and the error it raises
names this file.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get("EV_CONFIG") or Path(__file__).resolve().parents[1] / "config.yml"
)

_document: Optional[Dict[str, Any]] = None


def load(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """The parsed config file, read once and then kept.

    Read once because it is read from two modules and at several points in a
    query, and re-reading would mean a file edited mid-run took effect for some
    settings and not others. Restarting the service is the way to reload it,
    which is also when the containers it describes are started.
    """
    global _document
    if _document is not None:
        return _document

    _document = {}
    if not path.is_file():
        logger.info(f"no config file at {path}; no model container is configured")
        return _document

    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        logger.warning(
            f"PyYAML is not installed, so {path} cannot be read and no model "
            "container is configured. `pip install -r requirements.txt`."
        )
        return _document

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"could not read {path}: {exc}")
        return _document

    if parsed is None:
        return _document           # an empty file is a valid empty config
    if not isinstance(parsed, dict):
        logger.warning(f"{path} is not a mapping at the top level; ignoring it")
        return _document

    _document = parsed
    logger.info(f"configuration read from {path}: {sorted(parsed)}")
    return _document


def section(name: str) -> Dict[str, Any]:
    """One top-level mapping of the config file, or {} when it has none."""
    value = load().get(name)
    return value if isinstance(value, dict) else {}


def setting(name: str, key: str, env: str, default: Any) -> Any:
    """One value of one section: the environment, else the file, else `default`.

    An env var that is set but empty counts as unset, so `EV_CONTAINER_ARGS=`
    clears an override rather than becoming an empty setting that shadows the
    file.
    """
    override = os.environ.get(env)
    if override:
        return override
    value = section(name).get(key)
    return default if value is None else value


def as_args(value: Any) -> List[str]:
    """Container runtime arguments, however they were configured.

    A YAML list is already the argv this needs. A string is split the way a
    shell would, because an environment variable has no other way to carry more
    than one argument -- and because `--gpus all` is two.
    """
    if isinstance(value, str):
        return shlex.split(value)
    return [str(v) for v in (value or [])]
