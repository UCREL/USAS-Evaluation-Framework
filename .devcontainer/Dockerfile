# uv is copied from Astral's distroless image, which only holds the static
# /uv and /uvx binaries. Pin both the tag (for readability) and the digest
# (what Docker actually uses), and update them together.
ARG UV_VERSION=0.13.8
ARG UV_DIGEST=sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85
FROM ghcr.io/astral-sh/uv:${UV_VERSION}@${UV_DIGEST} AS uv

FROM ubuntu:latest


RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    git \
    make \
    wget \
    vim \
    build-essential \
    openssh-client \
    ca-certificates \
    gnupg2 \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

ARG USERNAME=ubuntu

USER $USERNAME
WORKDIR /home/$USERNAME


SHELL ["/bin/bash", "-c"]
COPY --from=uv --chown=$USERNAME:$USERNAME /uv /uvx /home/$USERNAME/.local/bin/

ENV PATH="/home/$USERNAME/.local/bin/:$PATH"

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv python install

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --no-install-workspace --all-extras