"""
Developer-only branch overlay for ML Forecast Lab.

A maintainer aid — **not** a user-facing feature. It lets a developer run
an arbitrary branch of the app's own GitHub repository in place of the
version baked into the add-on image, without rebuilding the image or
publishing a release. It is gated behind the ``developer_mode`` add-on
option (default ``false``); when that option is off, the System-tab card
and every endpoint here are inert, and the boot script ignores any
overlay that may be present on disk.

How it works
------------
The add-on image bakes the package at ``/app/ml_forecast_lab`` and boots
with ``PYTHONPATH=/app``. The runtime image has no ``git``, so this
module fetches the branch as a tarball over HTTPS (GitHub codeload),
extracts the ``ml_forecast_lab`` package into a persistent overlay
directory under ``/data`` (the only volume that survives a container
restart), and records an ``ACTIVE.json`` marker. The s6 boot script,
when ``developer_mode`` is on and the marker exists, prepends the
overlay to ``PYTHONPATH`` so the branch code shadows the bundled copy.

The bundled image is never modified. Reverting is just deleting the
overlay — or simply turning ``developer_mode`` off, which makes the boot
script ignore it on the next restart.

Security
--------
This is remote code execution by design: it downloads and runs code from
a branch. Three guards keep it maintainer-only:

1. ``developer_mode`` defaults off; the option is not surfaced in the
   normal config UI and nothing advertises this capability to users.
2. The source is locked to this repository (:data:`REPO_OWNER` /
   :data:`REPO_NAME`) — callers supply a branch name only, never a URL.
3. Branch names are validated against a strict charset, and tar members
   are extracted with path-traversal protection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tarfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The app's own repository. Hardcoded on purpose — the overlay must only
# ever run this project's branches, never an arbitrary URL.
REPO_OWNER = "psweens"
REPO_NAME = "ml-forecast-lab"

# Persistent overlay location. /data is the only volume that survives an
# add-on restart, so the boot script can re-apply the overlay each boot.
DEV_SRC_DIR = Path("/data/ml_forecast_lab/dev_src")
ACTIVE_MARKER = DEV_SRC_DIR / "ACTIVE.json"
PACKAGE_DIRNAME = "ml_forecast_lab"

# Branch names: GitHub allows a fairly broad charset, but we keep this
# strict (alnum plus . _ / -) and reject any "/.." traversal attempt.
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")


class DevBranchError(Exception):
    """Raised for any developer-overlay failure with a user-safe message."""


def developer_mode_enabled() -> bool:
    """Whether the ``developer_mode`` add-on option is on.

    The boot script exports the add-on option as ``DEVELOPER_MODE``
    (``"true"`` / ``"false"``), mirroring how ``LOG_LEVEL`` flows from
    bashio into the app. Defaults to off when the var is missing (e.g.
    local dev outside the add-on).
    """
    return str(os.environ.get("DEVELOPER_MODE", "false")).strip().lower() == "true"


def validate_branch(branch: str) -> str:
    """Return the branch name if valid, else raise :class:`DevBranchError`."""
    branch = (branch or "").strip()
    if not branch:
        raise DevBranchError("Branch name is empty.")
    if ".." in branch or branch.startswith("/") or branch.endswith("/"):
        raise DevBranchError(f"Invalid branch name: {branch!r}")
    if not _BRANCH_RE.match(branch):
        raise DevBranchError(
            f"Invalid branch name: {branch!r} "
            f"(allowed: letters, digits, '.', '_', '/', '-')."
        )
    return branch


def tarball_url(branch: str) -> str:
    """GitHub codeload tarball URL for a branch of this repo."""
    return (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
        f"/archive/refs/heads/{branch}.tar.gz"
    )


def commit_api_url(branch: str) -> str:
    """GitHub API URL to resolve a branch to its head commit SHA."""
    return (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        f"/commits/{branch}"
    )


def branches_api_url(page: int = 1, per_page: int = 100) -> str:
    """GitHub API URL listing branches of this repo (one page)."""
    return (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        f"/branches?per_page={per_page}&page={page}"
    )


def parse_branches(payload: Any) -> list[str]:
    """Extract sorted branch names from a GitHub branches-API payload.

    The default branch (``main``/``master``) is floated to the top; the
    rest are alphabetical. Pure and side-effect-free so it can be unit
    tested without touching the network.
    """
    names = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
    names = sorted(set(names), key=str.lower)

    def _key(name: str):
        return (name not in ("main", "master"), name.lower())

    return sorted(names, key=_key)


async def list_repo_branches(token: Optional[str] = None) -> list[str]:
    """Fetch all branch names for this repo from the GitHub API.

    Paginates up to a sane cap (GitHub returns 100/page). ``token`` is an
    optional GitHub token to raise the unauthenticated rate limit; the
    add-on has none by default, which is fine for occasional dev use.
    Raises :class:`DevBranchError` on any transport/HTTP failure so the
    caller can surface a clean message.
    """
    import aiohttp

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    names: list[str] = []
    max_pages = 10  # 1000 branches — far more than this project will ever have
    try:
        async with aiohttp.ClientSession() as session:
            for page in range(1, max_pages + 1):
                async with session.get(
                    branches_api_url(page=page),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 403:
                        raise DevBranchError(
                            "GitHub API rate limit hit while listing branches. "
                            "Wait a few minutes or type the branch name manually."
                        )
                    if resp.status != 200:
                        raise DevBranchError(
                            f"Could not list branches (HTTP {resp.status})."
                        )
                    page_items = await resp.json()
                if not page_items:
                    break
                names.extend(parse_branches(page_items))
                if len(page_items) < 100:
                    break
    except DevBranchError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DevBranchError(f"Could not list branches: {e}") from e

    # Re-sort the merged set so the default branch stays on top across pages.
    return parse_branches([{"name": n} for n in names])


def active_status() -> Optional[Dict[str, Any]]:
    """Return the active overlay's metadata, or ``None`` if none installed.

    Reads the on-disk marker regardless of ``developer_mode`` so the UI
    can distinguish "installed but disabled" from "not installed". The
    boot script — not this function — enforces that an overlay only takes
    effect while ``developer_mode`` is on.
    """
    try:
        if not ACTIVE_MARKER.is_file():
            return None
        data = json.loads(ACTIVE_MARKER.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:  # noqa: BLE001 — never let a bad marker crash a page
        logger.warning(f"Could not read dev overlay marker: {e}")
        return None


def is_overlay_running() -> bool:
    """Whether the *currently running* process booted from the overlay.

    True only when developer mode is on, a marker exists, and this
    module was imported from inside the overlay directory — i.e. the
    boot script actually applied the PYTHONPATH shadow. This is the
    honest "am I running branch code right now?" check used to label the
    version string, distinct from "an overlay is installed on disk".
    """
    if not (developer_mode_enabled() and ACTIVE_MARKER.is_file()):
        return False
    try:
        here = Path(__file__).resolve()
        return str(DEV_SRC_DIR.resolve()) in str(here)
    except Exception:  # noqa: BLE001
        return False


def version_label(base_version: str) -> str:
    """Annotate the base version with overlay info when running a branch.

    Returns e.g. ``"2.41.0 (dev: my-branch@1a2b3c4)"`` when booted from an
    overlay, or the unchanged ``base_version`` otherwise. Lets the version
    shown across the UI tell a developer they are off-release.
    """
    if not is_overlay_running():
        return base_version
    st = active_status() or {}
    branch = st.get("branch", "?")
    sha = st.get("sha_short") or (st.get("sha", "")[:7]) or "?"
    return f"{base_version} (dev: {branch}@{sha})"


def _safe_extract_package(raw_gz: bytes, dest_pkg_parent: Path) -> None:
    """Extract the repo tarball and place its ``ml_forecast_lab`` package.

    GitHub tarballs wrap everything in a top-level
    ``{repo}-{branch}/`` directory; the package we want lives at
    ``{top}/ml-forecast-lab/ml_forecast_lab/`` (addon dir → package). We
    locate the package by searching for the shallowest directory named
    ``ml_forecast_lab`` that contains both ``__init__.py`` and
    ``__main__.py`` — robust to repo-layout changes.

    Members are validated against path traversal before extraction.
    """
    try:
        tf = tarfile.open(fileobj=BytesIO(raw_gz), mode="r:gz")
    except Exception as e:  # noqa: BLE001
        raise DevBranchError(f"Downloaded data is not a valid .tar.gz: {e}") from e

    staging = dest_pkg_parent / "_extract_tmp"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        with tf:
            root = staging.resolve()
            for member in tf.getmembers():
                # Reject absolute paths and parent-dir escapes.
                target = (staging / member.name).resolve()
                if not str(target).startswith(str(root) + os.sep) and target != root:
                    raise DevBranchError(
                        f"Refusing unsafe tar member path: {member.name!r}"
                    )
                # Don't materialise device/fifo/link members.
                if member.isdev() or member.issym() or member.islnk():
                    continue
            tf.extractall(staging)  # members already validated above

        pkg = _find_package_dir(staging)
        if pkg is None:
            raise DevBranchError(
                "Branch tarball did not contain an 'ml_forecast_lab' package "
                "(expected at ml-forecast-lab/ml_forecast_lab)."
            )

        # Atomic-ish swap: stage the package next to the live dir, then
        # replace. Keeps a half-written overlay from ever being booted.
        final = dest_pkg_parent / PACKAGE_DIRNAME
        new = dest_pkg_parent / f"{PACKAGE_DIRNAME}.new"
        if new.exists():
            shutil.rmtree(new, ignore_errors=True)
        shutil.move(str(pkg), str(new))
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(new, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _find_package_dir(root: Path) -> Optional[Path]:
    """Find the shallowest valid ``ml_forecast_lab`` package under ``root``."""
    candidates = []
    for path in root.rglob(PACKAGE_DIRNAME):
        if (path / "__init__.py").is_file() and (path / "__main__.py").is_file():
            candidates.append(path)
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))


def install_from_tarball_bytes(
    branch: str,
    raw_gz: bytes,
    sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Install an overlay from already-downloaded tarball bytes.

    Split from the network fetch so it is trivially unit-testable with a
    synthetic tarball and no internet. Extracts the package, writes the
    ``ACTIVE.json`` marker, and returns the new status dict.
    """
    branch = validate_branch(branch)
    DEV_SRC_DIR.mkdir(parents=True, exist_ok=True)
    _safe_extract_package(raw_gz, DEV_SRC_DIR)

    status = {
        "branch": branch,
        "sha": sha or "",
        "sha_short": (sha or "")[:7],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{REPO_OWNER}/{REPO_NAME}",
    }
    tmp = ACTIVE_MARKER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, ACTIVE_MARKER)
    logger.warning(
        f"Developer overlay installed: branch={branch!r} sha={status['sha_short']!r}. "
        f"Restart the add-on to run it."
    )
    return status


def revert() -> bool:
    """Remove the overlay so the next boot uses the bundled image.

    Returns ``True`` if an overlay was present and removed, ``False`` if
    there was nothing to revert.
    """
    existed = DEV_SRC_DIR.exists()
    if existed:
        shutil.rmtree(DEV_SRC_DIR, ignore_errors=True)
        logger.warning(
            "Developer overlay removed; restart the add-on to return to the "
            "bundled release."
        )
    return existed
