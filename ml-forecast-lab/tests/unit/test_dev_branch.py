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
                       traversal=False, requirements=None) -> bytes:
    """Build a .tar.gz shaped like GitHub's codeload archive."""
    buf = io.BytesIO()
    top = f"ml-forecast-lab-{branch.replace('/', '-')}"
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name, data=b""):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        add(f"{top}/README.md", b"# repo\n")
        if requirements is not None:
            add(f"{top}/ml-forecast-lab/requirements.txt",
                requirements.encode("utf-8"))
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


# ---- overlay compatibility guard (the 2.42.2 anti-trap fix) ----

def test_overlay_compatible_true_when_nothing_installed(_isolate_overlay):
    assert dev_branch.overlay_is_compatible() is True


def test_overlay_incompatible_when_dev_branch_missing(_isolate_overlay):
    # The synthetic tarball ships __init__/__main__/models but no
    # dev_branch.py — i.e. a branch that predates developer mode.
    dev_branch.install_from_tarball_bytes("old", _make_repo_tarball("old"), sha="o")
    assert not (_isolate_overlay / "ml_forecast_lab" / "dev_branch.py").exists()
    assert dev_branch.overlay_is_compatible() is False


def test_overlay_compatible_when_dev_branch_present(_isolate_overlay):
    dev_branch.install_from_tarball_bytes("new", _make_repo_tarball("new"), sha="n")
    # Simulate a branch that carries the developer tooling.
    (_isolate_overlay / "ml_forecast_lab" / "dev_branch.py").write_text(
        "# dev tooling\n", encoding="utf-8",
    )
    assert dev_branch.overlay_is_compatible() is True


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


# ---- open-PR branch listing (dropdown = branches with an open PR) ----

def _pr(number, ref, full_name="psweens/ml-forecast-lab", title=None):
    return {
        "number": number,
        "title": title or f"PR {number}",
        "head": {"ref": ref, "repo": {"full_name": full_name}},
    }


def test_parse_pr_branches_same_repo_only_with_metadata():
    payload = [
        _pr(88, "claude/vibrant-babbage-7ypvor", title="Smart Setup"),
        _pr(12, "feature/x"),
        # Fork PR — head repo is someone else's; not overlayable, skipped.
        _pr(99, "patch-1", full_name="someone/ml-forecast-lab"),
    ]
    out = dev_branch.parse_pr_branches(payload)
    assert set(out) == {"claude/vibrant-babbage-7ypvor", "feature/x"}
    assert out["claude/vibrant-babbage-7ypvor"] == {"number": 88, "title": "Smart Setup"}


def test_parse_pr_branches_keeps_lowest_pr_number_per_branch():
    payload = [_pr(40, "shared"), _pr(7, "shared"), _pr(55, "shared")]
    out = dev_branch.parse_pr_branches(payload)
    assert out["shared"]["number"] == 7


def test_parse_pr_branches_skips_missing_repo_and_malformed():
    payload = [
        {"number": 1, "head": {"ref": "noinfo"}},          # no head.repo
        {"number": 2, "head": {"ref": "", "repo": {"full_name": "psweens/ml-forecast-lab"}}},
        "not-a-dict",
        {"number": 3, "head": {"ref": "ok", "repo": {"full_name": "psweens/ml-forecast-lab"}}},
    ]
    out = dev_branch.parse_pr_branches(payload)
    assert set(out) == {"ok"}


def test_parse_pr_branches_empty_and_non_list():
    assert dev_branch.parse_pr_branches([]) == {}
    assert dev_branch.parse_pr_branches(None) == {}
    assert dev_branch.parse_pr_branches({"head": {"ref": "x"}}) == {}


def test_compose_dev_branch_list_keeps_open_pr_and_default():
    all_branches = ["main", "claude/feat-a", "stale/merged", "claude/feat-b"]
    open_prs = {"claude/feat-a": {"number": 1}, "claude/feat-b": {"number": 2}}
    # Default always kept; stale (no PR) dropped; order preserved.
    assert dev_branch.compose_dev_branch_list(all_branches, open_prs) == [
        "main", "claude/feat-a", "claude/feat-b",
    ]


def test_compose_dev_branch_list_accepts_set_and_handles_empty():
    assert dev_branch.compose_dev_branch_list(["main", "x"], {"x"}) == ["main", "x"]
    # No open PRs → only the default branch survives.
    assert dev_branch.compose_dev_branch_list(["main", "x", "y"], {}) == ["main"]
    assert dev_branch.compose_dev_branch_list([], {"x"}) == []


def test_branch_is_closed_logic():
    open_prs = {"claude/feat-a": {"number": 1}}
    # Open PR → not closed.
    assert dev_branch.branch_is_closed("claude/feat-a", open_prs) is False
    # No PR → closed (eligible for auto-removal).
    assert dev_branch.branch_is_closed("claude/old", open_prs) is True
    # Default branch is never "closed".
    assert dev_branch.branch_is_closed("main", {}) is False
    assert dev_branch.branch_is_closed("master", {}) is False
    # No active branch installed → nothing to remove.
    assert dev_branch.branch_is_closed(None, open_prs) is False
    assert dev_branch.branch_is_closed("", open_prs) is False


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


# ---- dependency diff / install helpers ----

def test_requirement_package_name_parsing():
    f = dev_branch.requirement_package_name
    assert f('chronos-forecasting>=1.5.0,<3.0.0; platform_machine != "armv7l"') == "chronos-forecasting"
    assert f("granite_tsfm>=0.3.0") == "granite-tsfm"
    assert f("torch") == "torch"
    assert f("uvicorn[standard]==0.30") == "uvicorn"
    # Comments, blanks, and pip option/URL/path lines are ignored.
    assert f("# a comment") is None
    assert f("   ") is None
    assert f("-r other.txt") is None
    assert f("--extra-index-url https://x") is None
    assert f("git+https://github.com/x/y") is None


def test_new_requirements_returns_only_missing_full_lines():
    branch_reqs = (
        "# heavy deps\n"
        "torch>=2.0.0\n"
        'chronos-forecasting>=1.5.0,<3.0.0; platform_machine != "armv7l"\n'
        "granite-tsfm>=0.3.0,<1.0.0\n"
        "\n"
    )
    installed = {"torch", "numpy", "fastapi"}
    new = dev_branch.new_requirements(branch_reqs, installed=installed)
    # torch is already installed → skipped; the two foundation deps are new,
    # returned as full lines (markers/specifiers intact) for pip.
    assert new == [
        'chronos-forecasting>=1.5.0,<3.0.0; platform_machine != "armv7l"',
        "granite-tsfm>=0.3.0,<1.0.0",
    ]


def test_new_requirements_empty_when_all_satisfied_or_none():
    assert dev_branch.new_requirements(None) == []
    assert dev_branch.new_requirements("", installed=set()) == []
    assert dev_branch.new_requirements(
        "torch\nnumpy\n", installed={"torch", "numpy"}) == []


def test_dependency_install_supported_by_arch(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert dev_branch.dependency_install_supported() is True
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert dev_branch.dependency_install_supported() is True
    monkeypatch.setattr(platform, "machine", lambda: "armv7l")
    assert dev_branch.dependency_install_supported() is False


def test_install_captures_branch_requirements(_isolate_overlay):
    reqs = "torch>=2.0.0\nchronos-forecasting>=1.5.0\n"
    dev_branch.install_from_tarball_bytes(
        "feat", _make_repo_tarball("feat", requirements=reqs), sha="abc",
    )
    assert dev_branch.read_branch_requirements() == reqs
    # And the diff picks up only the genuinely-new line.
    new = dev_branch.new_requirements(
        dev_branch.read_branch_requirements(), installed={"torch"})
    assert new == ["chronos-forecasting>=1.5.0"]


def test_install_without_requirements_leaves_none(_isolate_overlay):
    dev_branch.install_from_tarball_bytes(
        "feat", _make_repo_tarball("feat"), sha="abc")
    assert dev_branch.read_branch_requirements() is None


def test_pip_install_command_targets_this_interpreter():
    import sys
    cmd = dev_branch.pip_install_command(["pkg-a", "pkg-b"])
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert cmd[-2:] == ["pkg-a", "pkg-b"]
