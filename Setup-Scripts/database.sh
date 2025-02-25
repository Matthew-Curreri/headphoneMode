#!/bin/bash

# Ensure the script runs with sudo privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root or with sudo."
    exit 1
fi

echo "Updating package lists..."
sudo apt update

# Install SQLite
echo "Installing SQLite..."
sudo apt install -y sqlite3

# Verify SQLite installation
if command -v sqlite3 &> /dev/null; then
    echo "SQLite installed successfully: $(sqlite3 --version)"
else
    echo "SQLite installation failed!"
    exit 1
fi

# Check if npm exists
if command -v npm &> /dev/null; then
    echo "npm is already installed: $(npm -v)"
else
    echo "npm not found. Checking for NVM..."

    # Check if NVM is installed
    if command -v nvm &> /dev/null; then
        echo "NVM found. Installing latest Node.js..."
        nvm install node
    else
        echo "NVM not found. Installing NVM first..."

        # Install NVM
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash

        # Load NVM in current shell
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

        # Install latest Node.js
        nvm install node
    fi

    # Verify npm installation
    if command -v npm &> /dev/null; then
        echo "npm installed successfully: $(npm -v)"
    else
        echo "npm installation failed!"
        exit 1
    fi
fi

echo "All dependencies installed successfully!"
