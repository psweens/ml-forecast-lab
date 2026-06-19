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
# The branch's requirements.txt is captured here on install so the
# installer can diff it against the running environment and install any
# new dependencies (see install-stream endpoint).
BRANCH_REQS_FILENAME = "branch_requirements.txt"

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


def pulls_api_url(page: int = 1, per_page: int = 100) -> str:
    """GitHub API URL listing OPEN pull requests of this repo (one page)."""
    return (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        f"/pulls?state=open&per_page={per_page}&page={page}"
    )


# The default branch is always offered in the dropdown even though it has
# no PR of its own — it's the "run mainline" choice and the revert target.
_DEFAULT_BRANCHES = ("main", "master")


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


def parse_pr_branches(payload: Any) -> Dict[str, Dict[str, Any]]:
    """Map each open PR's same-repo head branch → ``{number, title}``.

    Only PRs whose head lives in THIS repo are kept — a PR opened from a
    fork has a head ref that doesn't name a branch here, so codeload
    couldn't fetch it and offering it would only 404. When a branch backs
    more than one open PR (rare), the lowest PR number wins for a stable
    label. Pure and side-effect-free for unit testing.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, list):
        return out
    expect = f"{REPO_OWNER}/{REPO_NAME}".lower()
    for item in payload:
        if not isinstance(item, dict):
            continue
        head = item.get("head") or {}
        ref = head.get("ref")
        if not isinstance(ref, str) or not ref:
            continue
        # Skip fork PRs: include only when we can confirm the head repo is
        # this repo (absent repo info — e.g. a deleted fork — is skipped too,
        # since we can't safely codeload it).
        repo = head.get("repo") or {}
        full = repo.get("full_name")
        if not (isinstance(full, str) and full.lower() == expect):
            continue
        num = item.get("number")
        num = num if isinstance(num, int) else None
        title = item.get("title")
        prev = out.get(ref)
        if (prev is None or
                (num is not None and
                 (prev.get("number") is None or num < prev["number"]))):
            out[ref] = {
                "number": num,
                "title": title if isinstance(title, str) else None,
            }
    return out


def compose_dev_branch_list(
    all_branches: list,
    open_pr_branches: Any,
) -> list:
    """Filter existing branches to those backed by an open PR.

    Keeps the default branch (``main``/``master``) always available and
    preserves the incoming order (``list_repo_branches`` already floats the
    default to the top). ``open_pr_branches`` may be a dict (branch → PR
    meta) or any membership-testable collection of branch names.
    """
    keep = set(open_pr_branches or ())
    return [
        b for b in (all_branches or [])
        if b in keep or b in _DEFAULT_BRANCHES
    ]


def branch_is_closed(branch: Optional[str], open_pr_branches: Any) -> bool:
    """Whether an installed overlay's branch is now stale (no open PR).

    The default branch is never "closed" (it's always offered). Used to
    decide whether to auto-remove the overlay's files from the Pi.
    """
    if not branch:
        return False
    if branch in _DEFAULT_BRANCHES:
        return False
    return branch not in set(open_pr_branches or ())


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


async def list_open_pr_branches(
    token: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fetch the head branches of this repo's OPEN pull requests.

    Returns a dict mapping each same-repo head branch → ``{number, title}``
    so the System tab can list only branches with active work and annotate
    each with its PR. Paginates up to the same sane cap as the branch list.
    Raises :class:`DevBranchError` on any transport/HTTP failure so the
    caller can fall back cleanly (and, importantly, NOT auto-remove an
    overlay on a transient error).
    """
    import aiohttp

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out: Dict[str, Dict[str, Any]] = {}
    max_pages = 10  # 1000 open PRs — far more than this project will ever have
    try:
        async with aiohttp.ClientSession() as session:
            for page in range(1, max_pages + 1):
                async with session.get(
                    pulls_api_url(page=page),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 403:
                        raise DevBranchError(
                            "GitHub API rate limit hit while listing pull "
                            "requests. Wait a few minutes or type the branch "
                            "name manually."
                        )
                    if resp.status != 200:
                        raise DevBranchError(
                            f"Could not list pull requests (HTTP {resp.status})."
                        )
                    page_items = await resp.json()
                if not page_items:
                    break
                out.update(parse_pr_branches(page_items))
                if len(page_items) < 100:
                    break
    except DevBranchError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DevBranchError(f"Could not list pull requests: {e}") from e

    return out


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


def overlay_is_compatible() -> bool:
    """Whether an installed overlay carries the developer tooling.

    The boot script only applies an overlay whose package contains
    ``dev_branch.py``; an overlay of a branch that predates developer
    mode lacks it and is ignored at boot (otherwise it would shadow the
    revert UI and trap the user). The app mirrors that check so the
    System tab can explain *why* an installed overlay isn't running and
    point the user at Revert. Returns ``True`` when no overlay is
    installed (nothing to be incompatible).
    """
    if not ACTIVE_MARKER.is_file():
        return True
    return (DEV_SRC_DIR / PACKAGE_DIRNAME / "dev_branch.py").is_file()


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

        # Capture the branch's requirements.txt (sibling of the package's
        # parent — i.e. the addon dir) before the swap, so the installer can
        # diff it against the running env and install any new dependencies.
        reqs_src = pkg.parent / "requirements.txt"
        reqs_text = (
            reqs_src.read_text(encoding="utf-8", errors="replace")
            if reqs_src.is_file() else None
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

        # Persist (or clear) the captured requirements alongside the overlay.
        reqs_dest = dest_pkg_parent / BRANCH_REQS_FILENAME
        if reqs_text is not None:
            reqs_dest.write_text(reqs_text, encoding="utf-8")
        elif reqs_dest.exists():
            reqs_dest.unlink()
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


# ---------------------------------------------------------------------------
# Dependency installation for overlays that add new requirements.
#
# The overlay's PYTHONPATH shadow only swaps Python *source*; a branch that
# adds new packages (e.g. the foundation-model backends needing
# chronos-forecasting / granite-tsfm) also needs those installed. We diff
# the branch's captured requirements.txt against what's already installed
# in the running interpreter and install only the genuinely-new
# distributions into the live environment (not a --target dir), so pip
# reuses the image's existing packages and only fetches what's missing.
# Installs persist across add-on restarts (same container) but not across a
# rebuild — re-fetch the branch to reinstall. Skipped on 32-bit ARM, which
# has no wheels for the compiled transformers stack.
# ---------------------------------------------------------------------------

# pip requirement option lines (not packages) we never try to "install".
_REQ_OPTION_PREFIXES = ("-", "git+", "http://", "https://", ".", "/")


def canonical_name(name: str) -> str:
    """PEP 503 canonical distribution name (lower-case, dashes)."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def requirement_package_name(line: str) -> Optional[str]:
    """Extract the canonical package name from one requirements.txt line.

    Returns ``None`` for blank lines, comments, and pip option / URL /
    path lines (which we never install via the diff path).
    """
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(_REQ_OPTION_PREFIXES):
        return None
    # Drop environment markers and version specifiers; keep the name (and
    # strip any extras in brackets).
    head = re.split(r"[;<>=!~\[\( ]", line, 1)[0].strip()
    return canonical_name(head) if head else None


def read_branch_requirements() -> Optional[str]:
    """Return the installed overlay's captured requirements.txt text, if any."""
    path = DEV_SRC_DIR / BRANCH_REQS_FILENAME
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def installed_distribution_names() -> set:
    """Canonical names of every distribution installed in this interpreter."""
    import importlib.metadata as md
    names = set()
    for dist in md.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # noqa: BLE001
            name = None
        if name:
            names.add(canonical_name(name))
    return names


def new_requirements(
    branch_reqs_text: Optional[str],
    installed: Optional[set] = None,
) -> List[str]:
    """Return the branch requirement lines whose package isn't installed.

    Full original lines are returned (version specifiers and environment
    markers intact) so pip applies the branch's constraints and skips any
    line whose marker excludes this platform. Already-satisfied packages
    are left untouched, so core deps like torch/numpy are never disturbed.
    Pure and testable: pass ``installed`` to avoid touching the real env.
    """
    if not branch_reqs_text:
        return []
    if installed is None:
        installed = installed_distribution_names()
    out: List[str] = []
    seen = set()
    for raw in branch_reqs_text.splitlines():
        name = requirement_package_name(raw)
        if not name or name in installed or name in seen:
            continue
        seen.add(name)
        out.append(raw.split("#", 1)[0].strip())
    return out


def dependency_install_supported() -> bool:
    """Whether new compiled deps can be installed on this platform.

    False on 32-bit ARM (armv6l/armv7l), which has no wheels for the
    transformers/tokenizers stack — matching the ``platform_machine !=
    "armv7l"`` markers in requirements.txt.
    """
    import platform
    return platform.machine().lower() not in ("armv7l", "armv6l", "arm")


def pip_install_command(reqs: List[str]) -> List[str]:
    """Argv to install the given requirement specifiers into this env."""
    import sys
    return [
        sys.executable, "-m", "pip", "install",
        "--no-input", "--disable-pip-version-check", *reqs,
    ]
