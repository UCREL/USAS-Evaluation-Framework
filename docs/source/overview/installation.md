# Installation

The project uses [uv](https://docs.astral.sh/uv/). The Dev Container uses version `0.13.8`, but older versions from `0.9.6` onwards are also supported.

```bash
uv sync --all-extras
```

This installs the extras plus the `dev` and `docs` dependency groups. The `docs` group alone installs everything needed to build this documentation.
