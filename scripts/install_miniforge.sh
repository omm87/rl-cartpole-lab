#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  echo "Conda is already available: $(command -v conda)"
  exit 0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    PLATFORM="MacOSX"
    case "$ARCH" in
      arm64)   CONDA_ARCH="arm64" ;;
      x86_64)  CONDA_ARCH="x86_64" ;;
      *) echo "Unsupported macOS architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  Linux)
    PLATFORM="Linux"
    case "$ARCH" in
      x86_64)            CONDA_ARCH="x86_64" ;;
      aarch64|arm64)     CONDA_ARCH="aarch64" ;;
      *) echo "Unsupported Linux architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported operating system: $OS"
    exit 1
    ;;
esac

INSTALL_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
INSTALLER="/tmp/Miniforge3-${PLATFORM}-${CONDA_ARCH}.sh"
URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${PLATFORM}-${CONDA_ARCH}.sh"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download Miniforge. Install curl and run this script again."
  exit 1
fi

echo "Downloading Miniforge from:"
echo "  $URL"
curl -L "$URL" -o "$INSTALLER"

if [ -e "$INSTALL_DIR" ]; then
  echo "Installation directory already exists: $INSTALL_DIR"
  echo "Remove it or set MINIFORGE_DIR to a different path."
  exit 1
fi

bash "$INSTALLER" -b -p "$INSTALL_DIR"
rm -f "$INSTALLER"

"$INSTALL_DIR/bin/conda" init "$(basename "${SHELL:-bash}")" || true

echo
echo "Miniforge installed at: $INSTALL_DIR"
echo "For the current terminal session, run:"
echo "  source \"$INSTALL_DIR/etc/profile.d/conda.sh\""
echo "Then run:"
echo "  bash scripts/bootstrap.sh"
