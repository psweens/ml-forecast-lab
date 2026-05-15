# README_DESIGN.md

Phase 2 of the public-facing GitHub `README.md` refresh. Read after `README_SURVEY.md`.

## Headline recommendation: refine, don't rewrite

The current `README.md` is 146 lines and already implements most of what the brief asks for: logo + tagline opener, a focused install button, a "What this repository is" paragraph that frames who it's for, an honest architecture diagram tucked into `<details>`, a docs-routing table that prevents duplication, and a Development section. Tone is already dry-precise; British English is already consistent; there are no emojis or hype words to strip out.

What the existing README is **missing** against the brief:

1. **A Contributing section.** Nothing on the repo root says "PRs to area X are welcome, area Y needs scope agreement first." The current Support copy lives only in the per-add-on README.
2. **An explicit honest-status line.** The Support paragraph implies "first public release, best-effort", but doesn't put a one-line "Stable codebase, fresh to public, side-project cadence" up where evaluators see it.
3. **Acknowledgements.** No credit for the academic papers behind the 24 backends or the library authors the project leans on.
4. **A screenshot.** None exist in the repo. The brief allows leaving this out as long as we don't fake it with ASCII / Mermaid placeholders — we should add a single placeholder reference path that the author can drop a real PNG into post-merge.

What the existing README has that the brief might flag and we should **keep anyway**:

- **5 badges.** The brief defaults to fewer, but each badge here earns its slot (release / licence / tests / HA add-on / arch). The HA add-on badge is the one closest to decorative — but for a GitHub-discovery audience it signals "this is an HA add-on" at a glance, which is the question 60% of evaluators are asking. I recommend keeping all five and switching them from the static placeholder URLs (frozen while the repo was private) to the dynamic shields.io equivalents already commented in the source.
- **The `<details>`-collapsed ASCII architecture diagram.** It's already kept off the default scroll-budget. It earns its place when expanded — every block in the diagram corresponds to a real module in the survey.
- **The docs-routing table.** Critical for this repo because the README sits next to three other docs (per-add-on README, DOCS.md, CHANGELOG.md, MODEL_GUIDE.md) and readers need a clear "go here for that" map. This is not "table of contents" bloat; it's link disambiguation.

So Phase 3 will be a **surgical edit** of the existing `README.md`, not a rewrite.

## Voice & tone

**Pick: dry-precise**, matching the existing author voice with light wryness allowed but not forced.

Justification: the project is a 24-backend benchmarking add-on for HA power users who already have a Pi 5 and 30 days of recorder history. The audience is technical and self-selecting. The existing prose already commits to specifics over rhetoric ("If your fancy transformer can't beat naive, the issue isn't the model — it's the data"). Warm-welcoming would feel off; enthusiast would sound like marketing. Stay in the tone the existing docs already use — British English, specific numbers ("10–15 minutes on a Pi 5", "5 folds × 30 min"), no hype adjectives.

## Visual identity

- **Logo:** keep `logo.png` at `width="180"` in a centred `<div>` at the top, exactly as today. The 2127 × 2127 source is plenty for retina rendering at any practical size.
- **Hero / banner:** none warranted. Adding a full-width banner above the title would push the install button further down the scroll without conveying information the logo + tagline don't already convey. The brief explicitly tells us not to fabricate one.
- **Screenshot:** add **one** image reference near the top of "What it does", pointing to `docs/images/dashboard.png` (path to be created). If the file is missing the line breaks gracefully — but we'll flag it as the single "asset needed" deliverable rather than ship a placeholder graphic. The dashboard is the single highest-value evaluator screenshot.
- **Architecture diagram:** keep the existing ASCII inside `<details>`. Don't replace with Mermaid — the ASCII is more compact, renders identically everywhere, and the existing version is already accurate.
- **Palette in any new visuals:** if a diagram is regenerated, use the in-app palette from `style.css`: navy `#1a1a2e`, cyan `#00d4ff`, magenta `#e94560`. Badges stay on HA brand blue `#41bdf5` for consistency with the add-on store look — this is a deliberate mixed-palette choice the author already made.

## Badge policy

Keep **5 badges**, switching from static text to dynamic now that the repo is public:

| Badge | Why it earns its slot | Source |
|---|---|---|
| Latest release | Single most useful "is this maintained" signal | `img.shields.io/github/v/release/…` |
| Licence | Standard, lets evaluators check redistribution rights without scrolling | `img.shields.io/github/license/…` |
| Tests | Reinforces "this is a tested project" — 185 tests + workflow | `img.shields.io/github/actions/workflow/status/…/tests.yml` |
| HA add-on | Tells GitHub-discovery readers what *kind* of project this is in one glance | static `home-assistant` shield |
| Architectures | Lets the Pi 5 / amd64 / armv7 audience self-qualify in one second | static arch shield |

Rejected:

- Stargazers / forks: decorative, no information.
- "Made with PyTorch / FastAPI": stack badges add visual noise without resolving any evaluator question.
- Code coverage: real metric, but there's no public coverage report set up and adding one is out of scope.
- Downloads: HA add-on installs aren't counted by shields.io.

Action item for Phase 3: replace the five static-text URLs with the dynamic equivalents already commented in the existing README (lines 135–139). The comment explaining the freeze becomes obsolete and gets removed.

## Emoji policy

**None.** The existing docs use no emojis. The codebase comments use none. The Forecast Accuracy tab uses verdict *chips* (text), not emoji. Stay consistent.

## Section structure

Proposed order (asterisks mark new / restructured sections):

| # | Section | One-line justification |
|---|---|---|
| 1 | Logo + title + tagline + subhead + badges | Above-the-fold identity. Already in place. |
| 2 | **One-paragraph "what it is, who it's for"** — currently labelled "What this repository is" | Survives roughly as-is; the lead paragraph is already strong. Reframe heading. |
| 3 | **Screenshot** of the dashboard *(if produced; otherwise omitted, not faked)* | Evaluator-conversion. Asset needed. |
| 4 | Install | Two-line minimal path: one-click button + manual repo URL. Already in place. Link out for everything else. |
| 5 | What it does | Specific feature surface — 24 backends listed by family, what gets published to HA, what the workflow looks like. Already in place. |
| 6 | Architecture (`<details>`) | ASCII data-flow diagram. Already in place. |
| 7 | Documentation | The routing table — README → DOCS → MODEL_GUIDE → CHANGELOG. Already in place. |
| 8 | **Development** | Tests + CI gates. Refresh phrasing slightly; otherwise as-is. |
| 9 | **Contributing** *(new)* | Realistic asks: bug reports with version + log lines welcome; documentation gaps welcome; tested configurations for unfamiliar sensor types welcome; PRs into `main.py` orchestration need scope agreement first. Honest about what the maintainer can usefully review. Link to issue tracker + SECURITY.md. |
| 10 | **Status** *(new, one paragraph)* | "Stable codebase developed in private since [year], publicly released [version]. Maintained on a best-effort side-project cadence." |
| 11 | **Acknowledgements** *(new)* | Academic papers behind the 24 backends → link to MODEL_GUIDE which already names them; key libraries the project leans on (PyTorch, LightGBM, XGBoost, CatBoost, statsforecast, Optuna, pvlib, FastAPI, HTMX, Plotly); the `home-assistant/hassio-addons` base image. Five lines, no filler. |
| 12 | Licence | One line. Already in place. |

**Cut:** none from the existing README. **Add:** Contributing, Status, Acknowledgements. **Drop:** the now-obsolete "static while private" badge comment in the HTML.

## What goes in README vs what gets linked out

| In the README | Linked out to |
|---|---|
| Tagline, who it's for, minimal install path, the 24-backend headline, the published-sensor headline, the architecture diagram, the docs map, contributing summary, status, acknowledgements, licence | Full configuration reference → `ml-forecast-lab/DOCS.md` · Model picking → `docs/MODEL_GUIDE.md` · Release notes → `ml-forecast-lab/CHANGELOG.md` · Vulnerability reporting → `SECURITY.md` · First-experiment walkthrough → `ml-forecast-lab/README.md` (HA store **Info** tab) · Bug reports / discussion → GitHub issues URL |

**No new `CONTRIBUTING.md`.** The realistic contribution surface (3–5 paragraphs) fits inside the README and a separate file would be over-engineering for a side-project repo with one maintainer.

## Length budget check

Existing README is 146 lines. The three new sections (Contributing, Status, Acknowledgements) add roughly **20 lines** combined. Final length ≈ **165 lines**, roughly two screens on a laptop with the install section above the first fold. Still well under the brief's implicit budget.

## Assets needed but not in the repo

Only **one** is worth requesting before merge:

1. **`docs/images/dashboard.png`** — a single screenshot of the experiment dashboard with one experiment in production mode, the rank table or a forecast chart visible. Roughly 1600 × 900 px, PNG, the existing dark UI theme. Caption-free. This is the single asset that would meaningfully lift the README for the 60% of readers who are evaluating "is this real?".

Not required (worth having later, not blocking):

2. A second screenshot of the Forecast Accuracy tab.
3. A short GIF / MP4 of the Run-Pipeline → Promote → published-sensor loop.
4. A 1280 × 640 social-card / OG-image variant of the logo with the wordmark.

If `docs/images/dashboard.png` is not produced, the Phase 3 README will simply omit the image — no placeholder, no Mermaid stand-in.

---

**Awaiting approval before writing `README.md`.** Specifically, confirm:

- (a) refine-not-rewrite approach is OK,
- (b) the three new sections (Contributing / Status / Acknowledgements) are wanted,
- (c) whether to ship without the dashboard screenshot or hold Phase 3 until the asset exists.
