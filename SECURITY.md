# Security Policy

## Supported Versions

Only the current `main` branch is actively maintained. No long-term support is
provided for older commits or tagged releases at this time.

| Version / Branch | Supported |
|------------------|-----------|
| `main` (latest)  | ✅ Yes    |
| Older releases   | ❌ No     |

## Reporting a Vulnerability

**Please do not open a public GitHub issue to report a security vulnerability.**

To report a vulnerability privately, use one of the following methods:

- **GitHub private vulnerability reporting** (preferred):
  Navigate to
  [Security → Report a vulnerability](https://github.com/bibymaths/phoskintime/security/advisories/new)
  in this repository and submit a private advisory.
- **Email**: Send a description to `mishraabhinav36@gmail.com` with the subject
  line `[phoskintime] Security Vulnerability Report`.

## What to Include in a Report

A useful report includes:

- A clear description of the vulnerability and its potential impact.
- The affected file(s), module(s), or workflow(s).
- Steps to reproduce the issue, including any relevant commands or inputs.
- The version of Python, operating system, and Pixi/conda environment details.
- Any suggested remediation or patch, if available.

## Response

The maintainer will acknowledge receipt and assess the report. No fixed
response-time SLA is guaranteed, but reasonable effort will be made to respond
promptly. Progress updates will be communicated through the private advisory
channel.

## Scope

This security policy covers:

- Python source code in the repository.
- GitHub Actions workflows (`.github/workflows/`).
- Pixi environment and dependency configuration (`pixi.toml`, `pixi.lock`).
- Documentation deployment configuration.
- Reproducibility and configuration files.

## Out of Scope

- Third-party packages listed as dependencies. Please report vulnerabilities in
  dependencies directly to their respective maintainers.
- Issues in user-supplied input data or data pipelines outside this repository.
