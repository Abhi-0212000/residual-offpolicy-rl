# ============================================================================
# ResFiT – Residual Fine-Tuning with Off-Policy RL
# Multi-stage Docker build for isolated, reproducible training on shared servers
# ============================================================================
# Base: NVIDIA CUDA 12.8 + cuDNN 9 on Ubuntu 22.04 (matches dev environment)
# Python 3.10 via Miniforge (conda), all deps in a "residual" conda env
# ============================================================================

# ── Stage 1: Base system + conda ─────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # MuJoCo / rendering
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    # Disable interactive matplotlib backend
    MPLBACKEND=Agg

# System packages needed by MuJoCo, robosuite, OpenGL, ffmpeg, git, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget ca-certificates \
    build-essential cmake pkg-config \
    libgl1-mesa-dev libgles2-mesa-dev libegl1-mesa-dev libglfw3-dev \
    libglew-dev libglvnd-dev libglx-dev \
    libosmesa6-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev \
    unzip xvfb patchelf \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda + mamba) – small, fast, no Anaconda license issues
ENV CONDA_DIR=/opt/miniforge
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# Create the conda environment with Python 3.10 + ffmpeg
RUN conda create -n residual python=3.10 -y && \
    conda install -n residual -c conda-forge "ffmpeg>=6,<8" -y && \
    conda clean -afy

# Activate env for all subsequent RUN commands
SHELL ["conda", "run", "-n", "residual", "/bin/bash", "-c"]

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# Copy only dependency-related files first (Docker layer caching)
COPY resfit/lerobot/lerobot_requirements.txt /app/resfit/lerobot/lerobot_requirements.txt
COPY resfit/lerobot/setup_lerobot.sh /app/resfit/lerobot/setup_lerobot.sh
COPY resfit/dexmg/setup_dexmg.sh /app/resfit/dexmg/setup_dexmg.sh
COPY resfit/rl_finetuning/setup_rlpd_robosuite.sh /app/resfit/rl_finetuning/setup_rlpd_robosuite.sh

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

# Default entrypoint: activate conda env and run bash
# Users override CMD for specific training commands
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "residual"]
CMD ["/bin/bash"]
