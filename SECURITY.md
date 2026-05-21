# Security policy

## Supported versions

Only the latest minor version line (currently `2.34.x`) receives security
fixes. Older versions are unsupported — upgrade before reporting.

## Reporting a vulnerability

If you believe you have found a security issue in ML Forecast Lab,
please **do not open a public GitHub issue**. Instead, use GitHub's
[private vulnerability reporting][gh-report] on this repository
(*Security → Report a vulnerability*).

Include in your report:

- The affected version (`config.yaml`'s `version` field).
- A description of the issue and its potential impact.
- A minimal reproduction or proof-of-concept.
- Any suggested remediation, if you have one.

This is a free community app maintained on a best-effort basis. I
will aim to acknowledge a report within seven days and work toward a fix
within a reasonable timeframe relative to severity, but I make no
guarantees. Coordinated disclosure is appreciated.

## Scope

In scope:

- The app container, its Python code, and its web UI as shipped from
  this repository.
- The interaction between the app and the Home Assistant supervisor
  REST API.

Out of scope:

- Vulnerabilities in upstream dependencies — please report those to the
  respective project. Where the app can pin to a patched version, an
  issue here is welcome.
- Vulnerabilities in the user's own Home Assistant configuration,
  network, or host operating system.

## Disclosure

Once a fix is released, the changelog entry will note that it addresses
a security issue without disclosing the technical details until users
have had a reasonable window to upgrade.

[gh-report]: https://github.com/psweens/ml-forecast-lab/security/advisories/new
