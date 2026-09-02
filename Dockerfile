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
RUN uv lock && uv sync --active
WORKDIR /home/${USERNAME}

# Install specific variation of JAX, but don't add to prooject dependencies
RUN uv pip install jax[cuda]==0.8.1


###########################################################
FROM build-base AS build-dev

## Other tools
RUN curl -fsSL https://claude.ai/install.sh | bash


CMD ["echo", "Create!"]


###########################################################
# Same as build-dev, plus PyTorch, for cross-framework consistency checks
# (src/fabrique/models/qwen3vl/consistency_test.py, tests/qwen3vl_vision_parity.py).
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
RUN uv sync --active --group crosscheck
WORKDIR /home/${USERNAME}

CMD ["echo", "Create (with torch)!"]