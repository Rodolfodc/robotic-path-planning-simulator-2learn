"""Gera o index.html (página única, pronta para o GitHub Pages) a partir de
engine.js (motor) e template.html (visual).

    python web/build.py            # escreve ../index.html
    python web/build.py saida.html # escreve também a versão sem <!doctype>
"""

import sys
from pathlib import Path

aqui = Path(__file__).resolve().parent
motor = (aqui / "engine.js").read_text(encoding="utf-8")
# o motor entra inline, sem o bloco de exportação usado só no Node
motor = motor.replace('if (typeof module !== "undefined") module.exports = Engine;', "")
assert "</script>" not in motor

fragmento = (aqui / "template.html").read_text(encoding="utf-8").replace("/*ENGINE*/", motor)

cabecalho = (
    '<!doctype html>\n<html lang="pt-BR">\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '<meta name="description" content="Compara A* e campos potenciais no mesmo cenário aleatório, '
    'com um robô guiado por PID seguindo cada caminho.">\n'
)
destino = aqui.parent / "index.html"
destino.write_text(cabecalho + fragmento, encoding="utf-8")
print(f"escrito {destino} ({destino.stat().st_size} bytes)")

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(fragmento, encoding="utf-8")
    print(f"escrito {sys.argv[1]}")
