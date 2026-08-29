"""v2.52.1: the verdict headline must not claim checks the page marks ungraded.

During accuracy cold-start (no lead bucket has LEADBUCKET_MIN_N scored
samples, so accStatus is 'unknown') the headline logic filtered unknowns out
of [accStatus, covStatus, stabStatus] and, with the remaining chips good,
rendered the all-known wording "Looking healthy — forecasts are accurate,
bands are calibrated, runs agree." — asserting accuracy on the same page
whose Accuracy chip showed "—" and whose tile said "cold-start estimate".
It also produced verdict whiplash: a cold-start experiment with a provisional
error ratio past the poor threshold flipped from "Looking healthy" to
"Something looks off" the moment the tenth sample landed, with no behaviour
change. Source-contract pin (same idiom as test_external_comparison.
test_the_chart_reads_the_real_only_series): the partial-good branch must run
before the all-known wording and compose its clauses only from chips that are
individually graded.
"""

from pathlib import Path


def _headline_block():
    html = (Path(__file__).resolve().parents[2]
            / "ml_forecast_lab" / "web" / "templates" / "experiment.html").read_text()
    start = html.index("--- Overall headline ---")
    end = html.index("function _renderLeadTimeChart")
    return html[start:end]


class TestVerdictHeadlineContract:

    def test_all_unknown_wording_unchanged(self):
        assert ("'Warming up — first verdict available after a few "
                "production cycles.'") in _headline_block()

    def test_all_known_wording_unchanged(self):
        assert ("'Looking healthy — forecasts are accurate, bands are "
                "calibrated, runs agree.'") in _headline_block()

    def test_partial_branch_guards_the_all_known_claim(self):
        """The 'Looking healthy' claim may only render when all three chips
        are graded — the partial-good branch must be tested first."""
        block = _headline_block()
        assert "worst === 'good' && statuses.length < 3" in block, (
            "no partial-good branch: the all-known wording renders whenever "
            "the unknown-filtered worst is good, i.e. during cold-start"
        )
        assert (block.index("statuses.length < 3")
                < block.index("Looking healthy")), (
            "the partial-good guard must be evaluated before the "
            "unconditional all-known wording"
        )

    def test_partial_clauses_come_only_from_graded_chips(self):
        """Each clause of the composed headline is pushed by its own chip's
        status — never inherited from the filtered aggregate."""
        block = _headline_block()
        for chip, clause in [
            ("accStatus", "forecasts are accurate"),
            ("covStatus", "bands are calibrated"),
            ("stabStatus", "runs agree"),
        ]:
            assert ("if (" + chip + " === 'good') "
                    "goodClauses.push('" + clause + "');") in block

    def test_partial_names_what_is_still_warming_up(self):
        block = _headline_block()
        for chip, name in [
            ("accStatus", "accuracy"),
            ("covStatus", "calibration"),
            ("stabStatus", "stability"),
        ]:
            assert ("if (" + chip + " === 'unknown') "
                    "warming.push('" + name + "');") in block
        assert "still warming up.'" in block
