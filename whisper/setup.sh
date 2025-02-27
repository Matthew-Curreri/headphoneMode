#!/usr/bin/env bash
#
# Script Name: install_whisper_turbo.sh
# Description: Installs dependencies and Whisper in the 'whisper' conda environment, 
#              ensuring PyTorch is installed with CUDA support, ffmpeg is installed, 
#              and the turbo model is downloaded.
#
# Usage:       ./install_whisper_turbo.sh
#
# NOTE: This script assumes you have conda installed and accessible on PATH.
#       If "conda" commands are not recognized, you may need to adjust your shell
#       configuration or the script to properly load conda.

set -e

echo ">>> Activating conda environment 'whisper'..."
# Ensure the conda command is available within this script
eval "$(conda shell.bash hook)"

conda activate whisper || {
  echo "ERROR: Could not activate the 'whisper' environment. Make sure it exists."
  exit 1
}

echo ">>> Checking if PyTorch is installed..."
if ! python -c "import torch" &>/dev/null; then
  echo ">>> PyTorch not found. Installing PyTorch with CUDA 11.3 support..."
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
else
  echo ">>> PyTorch is already installed."
fi

echo ">>> Installing/Updating Whisper from GitHub..."
# IMPORTANT: Removed --no-deps so that dependencies like 'tqdm' are installed
pip install --upgrade --force-reinstall git+https://github.com/openai/whisper.git

echo ">>> Installing ffmpeg via apt (Ubuntu/Debian-based) as an example..."
# Uncomment or modify for your OS
sudo apt update && sudo apt install -y ffmpeg

# If you encounter Rust-related build errors while installing tiktoken,
# uncomment the following lines (and adjust for your OS) to install Rust:
# echo ">>> Installing Rust environment..."
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# source $HOME/.cargo/env

# For some platforms, you may need setuptools-rust:
# pip install setuptools-rust

echo ">>> Downloading the 'turbo' model to ensure it's cached locally..."
python -c "import whisper; whisper.load_model('turbo')"

echo ">>> Installation and model download complete!"
exit 0

