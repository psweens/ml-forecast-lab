"""Unit tests for the developer-mode branch overlay (ml_forecast_lab.dev_branch).

The network fetch lives in the web endpoint; the testable core here is
branch validation, tarball extraction (incl. path-traversal defence),
marker read/write, revert, and the version label. A synthetic in-memory
tarball mirroring GitHub's codeload layout exercises install without any
network.
"""

import io
import json
import tarfile

import pytest

from ml_forecast_lab import dev_branch


@pytest.fixture(autouse=True)
def _isolate_overlay(tmp_path, monkeypatch):
    """Redirect the overlay paths into a tmp dir for every test."""
    dev_src = tmp_path / "dev_src"
    monkeypatch.setattr(dev_branch, "DEV_SRC_DIR", dev_src)
    monkeypatch.setattr(dev_branch, "ACTIVE_MARKER", dev_src / "ACTIVE.json")
    return dev_src


def _make_repo_tarball(branch="my-branch", with_package=True,
                       traversal=False) -> bytes:
    """Build a .tar.gz shaped like GitHub's codeload archive."""
    buf = io.BytesIO()
    top = f"ml-forecast-lab-{branch.replace('/', '-')}"
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name, data=b""):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        add(f"{top}/README.md", b"# repo\n")
        if with_package:
            pkg = f"{top}/ml-forecast-lab/ml_forecast_lab"
            add(f"{pkg}/__init__.py", b"__version__='9.9.9'\n")
            add(f"{pkg}/__main__.py", b"# entry\n")
            add(f"{pkg}/models/__init__.py", b"# models\n")
        if traversal:
            add("../evil.py", b"pwned\n")
    return buf.getvalue()


# ---- branch validation ----

@pytest.mark.parametrize("branch", [
    "main", "claude/my-feature", "v2.42.0", "feature_1", "a-b.c/d",
])
def test_validate_branch_accepts_valid(branch):
    assert dev_branch.validate_branch(branch) == branch


@pytest.mark.parametrize("branch", [
    "", "   ", "../etc/passwd", "/leading", "trailing/",
    "has space", "semi;colon", "pipe|x", "back`tick",
])
def test_validate_branch_rejects_invalid(branch):
    with pytest.raises(dev_branch.DevBranchError):
        dev_branch.validate_branch(branch)


# ---- developer_mode gate ----

def test_developer_mode_reads_env(monkeypatch):
    monkeypatch.setenv("DEVELOPER_MODE", "true")
    assert dev_branch.developer_mode_enabled() is True
    monkeypatch.setenv("DEVELOPER_MODE", "false")
    assert dev_branch.developer_mode_enabled() is False
    monkeypatch.delenv("DEVELOPER_MODE", raising=False)
    assert dev_branch.developer_mode_enabled() is False


# ---- install / status / revert ----

def test_install_extracts_package_and_writes_marker(_isolate_overlay):
    raw = _make_repo_tarball(branch="claude/feat")
    status = dev_branch.install_from_tarball_bytes(
        "claude/feat", raw, sha="1a2b3c4d5e6f",
    )
    pkg = _isolate_overlay / "ml_forecast_lab"
    assert (pkg / "__init__.py").is_file()
    assert (pkg / "__main__.py").is_file()
    assert (pkg / "models" / "__init__.py").is_file()
    assert status["branch"] == "claude/feat"
    assert status["sha_short"] == "1a2b3c4"

    on_disk = json.loads((_isolate_overlay / "ACTIVE.json").read_text())
    assert on_disk["branch"] == "claude/feat"
    assert dev_branch.active_status()["sha"] == "1a2b3c4d5e6f"


def test_install_reinstall_replaces_previous(_isolate_overlay):
    dev_branch.install_from_tarball_bytes("a", _make_repo_tarball("a"), sha="aaa")
    dev_branch.install_from_tarball_bytes("b", _make_repo_tarball("b"), sha="bbb")
    assert dev_branch.active_status()["branch"] == "b"
    # No staging leftovers.
    assert not (_isolate_overlay / "_extract_tmp").exists()
    assert not (_isolate_overlay / "ml_forecast_lab.new").exists()


def test_install_rejects_non_gzip(_isolate_overlay):
    with pytest.raises(dev_branch.DevBranchError):
        dev_branch.install_from_tarball_bytes("x", b"not a tarball", sha="x")


def test_install_rejects_tarball_without_package(_isolate_overlay):
    raw = _make_repo_tarball(with_package=False)
    with pytest.raises(dev_branch.DevBranchError, match="ml_forecast_lab"):
        dev_branch.install_from_tarball_bytes("x", raw, sha="x")


def test_install_rejects_path_traversal_member(_isolate_overlay):
    raw = _make_repo_tarball(traversal=True)
    with pytest.raises(dev_branch.DevBranchError, match="unsafe"):
        dev_branch.install_from_tarball_bytes("x", raw, sha="x")
    # Nothing escaped the overlay dir.
    assert not (_isolate_overlay.parent / "evil.py").exists()


def test_active_status_none_when_not_installed(_isolate_overlay):
    assert dev_branch.active_status() is None


def test_revert_removes_overlay(_isolate_overlay):
    dev_branch.install_from_tarball_bytes("a", _make_repo_tarball("a"), sha="aaa")
    assert dev_branch.active_status() is not None
    assert dev_branch.revert() is True
    assert dev_branch.active_status() is None
    assert not _isolate_overlay.exists()


def test_revert_noop_when_nothing_installed(_isolate_overlay):
    assert dev_branch.revert() is False


# ---- branch listing (parse only; network lives in the endpoint) ----

def test_parse_branches_floats_default_to_top():
    payload = [
        {"name": "zeta"},
        {"name": "claude/feature"},
        {"name": "main"},
        {"name": "Alpha"},
    ]
    assert dev_branch.parse_branches(payload) == [
        "main", "Alpha", "claude/feature", "zeta",
    ]


def test_parse_branches_master_also_floats():
    assert dev_branch.parse_branches(
        [{"name": "x"}, {"name": "master"}]
    ) == ["master", "x"]


def test_parse_branches_dedupes_and_ignores_malformed():
    payload = [
        {"name": "dup"}, {"name": "dup"}, {"name": 123},
        {"nope": "x"}, "string-not-dict", {"name": "abc"},
    ]
    assert dev_branch.parse_branches(payload) == ["abc", "dup"]


def test_parse_branches_empty_and_non_list():
    assert dev_branch.parse_branches([]) == []
    assert dev_branch.parse_branches(None) == []
    assert dev_branch.parse_branches({"name": "x"}) == []


# ---- version label ----

def test_version_label_plain_when_not_running(monkeypatch):
    monkeypatch.setattr(dev_branch, "is_overlay_running", lambda: False)
    assert dev_branch.version_label("2.42.0") == "2.42.0"


def test_version_label_annotated_when_running(_isolate_overlay, monkeypatch):
    dev_branch.install_from_tarball_bytes(
        "claude/feat", _make_repo_tarball("claude/feat"), sha="1a2b3c4d",
    )
    monkeypatch.setattr(dev_branch, "is_overlay_running", lambda: True)
    assert dev_branch.version_label("2.42.0") == "2.42.0 (dev: claude/feat@1a2b3c4)"
