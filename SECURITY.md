# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.x     | ✅        |
| < 1.0   | ❌        |

## Scope

`mplmasterpro` is a plotting library: it reads data you already have and
writes image files. It does not open network connections, execute remote
code or handle credentials. The Docker image and Streamlit viewer are
intended for local use; do not expose them to untrusted networks without
adding authentication (the Jupyter server in the image runs token-less by
design for local convenience).

## Reporting a vulnerability

Please email **satvikpraveen707@gmail.com** with a description of the issue,
steps to reproduce and the affected version. You will receive an
acknowledgement within 72 hours. Please do not open a public issue for
security-sensitive reports until a fix is available.

## Dependency hygiene

Dependencies are monitored monthly by Dependabot (`.github/dependabot.yml`)
and the test matrix runs against the latest releases of Matplotlib, NumPy,
pandas and SciPy on every push.
