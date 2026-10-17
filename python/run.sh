#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Creates the virtual environment on the first run
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    if ! python3 -m venv .venv; then
        rm -rf .venv
        echo
        echo "Failed to create the venv. On Ubuntu/Debian install it with:"
        echo "    sudo apt install python3-venv"
        exit 1
    fi
fi

# Installs the dependencies only once (marker created after success)
if [ ! -f .venv/.installed ]; then
    echo "Installing dependencies..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/.installed
fi

# The Matplotlib window uses Tk, which comes from the system, not from pip
if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    echo
    echo "Tkinter not found (required for the window). Install it with:"
    echo "    Ubuntu/Debian: sudo apt install python3-tk"
    echo "    Fedora:        sudo dnf install python3-tkinter"
    echo "    Arch:          sudo pacman -S tk"
    exit 1
fi

.venv/bin/python interface_astar.py
