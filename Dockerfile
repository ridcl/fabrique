# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-base
FROM ubuntu:24.04 AS build-base
RUN userdel -r ubuntu


## Basic system setup

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND=noninteractive \
    TERM=linux

ENV TERM=xterm-color

ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8

RUN apt-get update && apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gpg \
        gpg-agent \
        less \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        llvm \
        locales \
        tk-dev \
        tzdata \
        unzip \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        zstd \
    && sed -i "s/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g" /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt clean

## System packages

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    python-is-python3 \
    python3 \
    python3-pip \
    && apt-get clean \
    && pip install uv --break-system-packages

## Add user & enable sudo

ARG USERNAME=devpod
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID ${USERNAME} \
    && useradd --uid $USER_UID --gid $USER_GID -ms /bin/bash ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && apt-get install -y sudo \
    && apt-get clean \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && echo 'export PATH=${PATH}:~/.local/bin' >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /home/${USERNAME}

## Python packages

# avoid uv warning related to Windows/Linux compatibility issues
ENV UV_LINK_MODE=copy

# create globally visible venv
# also set $VIRTUAL_ENV which will be used by uv
ENV VIRTUAL_ENV=/venv
RUN sudo mkdir "$VIRTUAL_ENV" \
    && sudo chown -R ${USERNAME}:${USERNAME} "$VIRTUAL_ENV"

# install the project
ENV BUILD_DIR=/app
COPY --chown=${USERNAME}:${USERNAME} . "$BUILD_DIR"


WORKDIR "${BUILD_DIR}"
# --locked installs exactly what the committed uv.lock says and FAILS if the lock
# has drifted from pyproject.toml, instead of quietly re-resolving.  That is the
# whole reproducibility story: pyproject.toml carries open ranges so fabrique
# stays installable as a library, and the lock carries the exact versions.
# (The old `uv lock && uv sync` re-resolved against latest PyPI on every build,
# which is how jaxlib 0.11.1 kept getting in.  Re-lock deliberately with
# `uv lock` on the host and commit the diff.)
#
# --extra cuda12 replaces the former out-of-band `uv pip install jax[cuda12]`.
# Because the extra is declared in pyproject.toml it is now IN the lock, so
# nothing lives outside it and no --inexact workaround is needed downstream.
#
# --no-install-project: the devcontainer bind-mounts the real workspace over
# /workspaces/fabrique and puts `src` on PYTHONPATH, so the code comes from
# there.  Installing the project here would bake an editable install pointing at
# /app/src -- a build-time snapshot that silently serves stale code whenever
# PYTHONPATH is not set.  Dependencies are what we want from the image.
RUN uv sync --active --locked --extra cuda12 --no-install-project
WORKDIR /home/${USERNAME}


###########################################################
FROM build-base AS build-dev

## Other tools
RUN curl -fsSL https://claude.ai/install.sh | bash


CMD ["echo", "Create!"]


###########################################################
# Same as build-dev, plus PyTorch, for cross-framework consistency checks
# (crosscheck/qwen3vl_consistency.py, crosscheck/qwen3vl_vision_parity.py).
# Select it from .devcontainer/torch/devcontainer.json.
#
# The `crosscheck` group pins the *CPU* build of torch on purpose: the CUDA
# build depends on nvidia-*-cu13 wheels which install over JAX's nvidia-*-cu12
# ones at the same paths (libcudnn.so.9 among them), leaving JAX loading a
# CUDA-13 cuDNN.  That fails every cuDNN flash-attention shape check at runtime
# and silently falls back to the XLA kernel, which materialises an O(seq_len^2)
# logits buffer.  The CPU build has no nvidia dependencies, and the consistency
# checks run on CPU regardless (loading two copies of a model is memory-bound).
FROM build-dev AS build-dev-torch

# ARG scope does not cross FROM, so USERNAME has to be redeclared here
# (BUILD_DIR is an ENV and does carry over).
ARG USERNAME=devpod

WORKDIR "${BUILD_DIR}"
# Repeat --extra cuda12: `uv sync` makes the environment match the request
# exactly, so omitting it here would UNINSTALL the CUDA jax that build-base
# installed and leave a CPU-only environment.  Previously this needed --inexact
# because jax[cuda12] lived outside the lock; now that it is a declared extra,
# naming it is enough and the environment stays fully lock-governed.
RUN uv sync --active --locked --extra cuda12 --group crosscheck --no-install-project
WORKDIR /home/${USERNAME}

CMD ["echo", "Create (with torch)!"]