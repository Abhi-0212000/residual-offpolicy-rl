# ============================================================================
# ResFiT – Residual Fine-Tuning with Off-Policy RL
# Multi-stage Docker build for isolated, reproducible training on shared servers
# ============================================================================
# Base: NVIDIA CUDA 12.8 + cuDNN 9 on Ubuntu 22.04 (matches dev environment)
# Python 3.10 from system + pip (no conda — Docker IS the isolation)
# ============================================================================

# ── Stage 1: Base system ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # MuJoCo / rendering
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    # Disable interactive matplotlib backend
    MPLBACKEND=Agg \
    # pip should not warn about running as root
    PIP_ROOT_USER_ACTION=ignore

# System packages: Python 3.10, MuJoCo deps, OpenGL, FFmpeg, git, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git git-lfs curl wget ca-certificates \
    build-essential cmake pkg-config \
    libgl1-mesa-dev libgles2-mesa-dev libegl1-mesa-dev libglfw3-dev \
    libglew-dev libglvnd-dev libglx-dev \
    libosmesa6-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev \
    unzip xvfb patchelf \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/python3, and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# Copy only dependency-related files first (Docker layer caching)
COPY resfit/lerobot/lerobot_requirements.txt /app/resfit/lerobot/lerobot_requirements.txt

# ── Install PyTorch first (biggest layer, cached unless CUDA version changes)
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Clone & install deps repos (lerobot, robosuite, dexmimicgen, mimicgen)
# lerobot
RUN mkdir -p /app/deps && \
    git clone https://github.com/huggingface/lerobot.git /app/deps/lerobot && \
    git -C /app/deps/lerobot checkout 69901b9b6a2300914ca3de0ea14b6fa6e0203bd4 && \
    pip install --no-cache-dir -e /app/deps/lerobot --no-deps

# lerobot requirements + torchcodec
RUN pip install --no-cache-dir \
    -r /app/resfit/lerobot/lerobot_requirements.txt && \
    pip install --no-cache-dir torchcodec --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir datasets==3.6.0

# robosuite
RUN git clone https://github.com/ARISE-Initiative/robosuite /app/deps/robosuite && \
    git -C /app/deps/robosuite checkout 77a4751233c29456a5381209e30dd0dbf39a6557 && \
    pip install --no-cache-dir -e /app/deps/robosuite

# dexmimicgen
RUN git clone https://github.com/NVlabs/dexmimicgen.git /app/deps/dexmimicgen && \
    git -C /app/deps/dexmimicgen checkout e606f36a38b1d4ba8f56d06d6c0cd059b20ebbaf && \
    pip install --no-cache-dir -e /app/deps/dexmimicgen

# mimicgen
RUN git clone https://github.com/NVlabs/mimicgen.git /app/deps/mimicgen && \
    git -C /app/deps/mimicgen checkout main && \
    pip install --no-cache-dir -e /app/deps/mimicgen

# ── Remaining pip packages (from SETUP_FIXES.md + setup scripts)
RUN pip install --no-cache-dir \
    mink==0.0.7 \
    gymnasium==1.1.1 \
    PyOpenGL==3.1.10 \
    PyOpenGL-accelerate==3.1.10 \
    ipdb serial deepdiff tabulate \
    mujoco==3.3.2 \
    protobuf==3.20.3 \
    diffusers==0.33.1 \
    draccus==0.10.0 \
    torchrl==0.11.1 tensordict==0.11.0 \
    wandb einops psutil \
    hydra-core omegaconf \
    huggingface-hub safetensors \
    h5py scipy matplotlib pillow \
    "numba>=0.60" "llvmlite>=0.44"

# Verify critical imports
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "from torchrl._torchrl import SumSegmentTreeFp32; print('torchrl C++ OK')" && \
    python -c "import mujoco; print(f'MuJoCo {mujoco.__version__}')" && \
    python -c "import robosuite; print('robosuite OK')" && \
    python -c "import wandb; print(f'wandb {wandb.__version__}')"

# ── Stage 3: Final image with project code ───────────────────────────────────
FROM deps AS final

WORKDIR /app

# Copy the entire project (respects .dockerignore)
COPY . /app/

# Add project root to PYTHONPATH (needed for `import resfit`)
ENV PYTHONPATH="/app:$PYTHONPATH"

CMD ["/bin/bash"]
