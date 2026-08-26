"""The advertised backend count tracks the wired-in registry (v2.51.x).

The number of model backends appears in prose across the READMEs and DOCS —
it is a headline feature — and it had silently drifted twice: a smoke-test
docstring and the dry-run pipeline both said 24 while the READMEs said 29.
Prose cannot be computed, so it is guarded instead: ``models.WIRED_BACKENDS``
(the static import map, independent of which optional deps are installed) is
the single source of truth, and every "N backends" claim in the user-facing
docs must equal its length. Adding or removing a backend now fails this test
until the docs are updated in the same change.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ml_forecast_lab.models import WIRED_BACKENDS

APP_DIR = Path(__file__).resolve().parents[2]

# User-facing documents that advertise the count. The repo-root README sits
# one level above the app directory; it is skipped when absent (e.g. the app
# directory checked out standalone) rather than failed.
DOC_PATHS = [
    APP_DIR.parent / "README.md",
    APP_DIR / "README.md",
    APP_DIR / "DOCS.md",
]

COUNT_CLAIM = re.compile(r"\b(\d+)\s+(?:model\s+)?backends\b", re.IGNORECASE)


def _claims(path: Path):
    return [
        (int(m.group(1)), m.group(0))
        for m in COUNT_CLAIM.finditer(path.read_text(encoding="utf-8"))
    ]


class TestBackendCountStaysHonest:
    def test_wired_backend_map_is_nonempty(self):
        assert len(WIRED_BACKENDS) >= 20, (
            "the wired-backend map shrank drastically — is the import table "
            "in models/__init__.py intact?"
        )

    @pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
    def test_every_advertised_count_matches_the_wired_map(self, path):
        if not path.exists():
            pytest.skip(f"{path} not present in this checkout")
        claims = _claims(path)
        assert claims, (
            f"{path.name} no longer mentions a backend count — if that is "
            f"deliberate, remove it from DOC_PATHS"
        )
        wrong = [c for c in claims if c[0] != len(WIRED_BACKENDS)]
        assert not wrong, (
            f"{path.name} advertises {wrong} but models/__init__.py wires "
            f"{len(WIRED_BACKENDS)} backends. Update the prose in the same "
            f"change that added or removed a backend."
        )

    def test_no_source_docstring_pins_a_count(self):
        """Code and test docstrings must count dynamically or not at all —
        the two stale '24 backends' docstrings this test buries."""
        for rel in ("tests/smoke/test_harness.py", "tests/dryrun_pipeline.py"):
            text = (APP_DIR / rel).read_text(encoding="utf-8")
            assert not COUNT_CLAIM.search(text), (
                f"{rel} pins a literal backend count; phrase it as "
                f"'every registered backend' or derive it from the registry"
            )
