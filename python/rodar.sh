#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Cria o ambiente virtual na primeira execucao
if [ ! -x .venv/bin/python ]; then
    echo "Criando ambiente virtual..."
    if ! python3 -m venv .venv; then
        rm -rf .venv
        echo
        echo "Falha ao criar o venv. No Ubuntu/Debian instale com:"
        echo "    sudo apt install python3-venv"
        exit 1
    fi
fi

# Instala as dependencias so uma vez (marcador criado apos sucesso)
if [ ! -f .venv/.instalado ]; then
    echo "Instalando dependencias..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/.instalado
fi

# A janela do Matplotlib usa Tk, que vem do sistema e nao do pip
if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    echo
    echo "Tkinter nao encontrado (necessario para a janela). Instale com:"
    echo "    Ubuntu/Debian: sudo apt install python3-tk"
    echo "    Fedora:        sudo dnf install python3-tkinter"
    echo "    Arch:          sudo pacman -S tk"
    exit 1
fi

.venv/bin/python interface_astar.py
