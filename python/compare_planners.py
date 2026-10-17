"""
Compara Potential Fields (planejador local/reativo) com A* (planejador global)
no MESMO cenário aleatório (início, objetivo e obstáculos idênticos).

Reaproveita os scripts existentes:
  - potential_fields.py : gerar_obstaculos(), planejar()
  - astar.py            : a_estrela(), OBSTACULO

Uso: python compare_planners.py [semente] [--quiet]
  --quiet : silencia o log das forças do potential_fields

Também pode ser importado (ver follow_path.py): criar_cenario(), rodar_pf(),
rodar_astar(), comprimento().
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import potential_fields as pf
from astar import a_estrela, OBSTACULO

INICIO = np.array([0.0, 0.0])
OBJETIVO = np.array([10.0, 10.0])
N_OBSTACULOS = 4

res = 0.2          # tamanho da célula da grade do A*
margem = 0.1       # margem de segurança ao redor dos obstáculos


# ---------------------------------------------------------------
# 1. Cenário (gerado pelo potential_fields.py)
# ---------------------------------------------------------------
def criar_cenario(semente=None):
    """Retorna (semente, obstaculos). Mesma semente -> mesmo cenário."""
    if semente is None:
        semente = int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(semente)
    return semente, pf.gerar_obstaculos(N_OBSTACULOS, rng, INICIO, OBJETIVO)


# ---------------------------------------------------------------
# 2. Grade para o A* (rasteriza os mesmos círculos)
#    mapa[y, x]: peso 1 = livre, OBSTACULO = bloqueada
# ---------------------------------------------------------------
def criar_mapa(obstaculos):
    n = int(round(10 / res)) + 1
    xs = np.arange(n) * res
    X, Y = np.meshgrid(xs, xs)                   # X[y, x], Y[y, x]
    mapa = np.ones((n, n))
    for centro, raio in obstaculos:
        mapa[np.hypot(X - centro[0], Y - centro[1]) <= raio + margem] = OBSTACULO
    return mapa


def comprimento(caminho):
    caminho = np.asarray(caminho)
    return float(np.sum(np.linalg.norm(np.diff(caminho, axis=0), axis=1)))


# ---------------------------------------------------------------
# 3. Execução dos planejadores. Ambos retornam (caminho Nx2, chegou)
# ---------------------------------------------------------------
def rodar_pf(obstaculos):
    return pf.planejar(INICIO, OBJETIVO, obstaculos)


def rodar_astar(mapa):
    cel_ini = tuple(np.round(INICIO / res).astype(int))
    cel_obj = tuple(np.round(OBJETIVO / res).astype(int))
    cam, _ = a_estrela(mapa, cel_ini, cel_obj)
    if cam is None:
        return np.array([INICIO]), False
    return np.array(cam) * res, True


# ---------------------------------------------------------------
# 4. Programa principal: executa, imprime métricas e plota
# ---------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    semente, obstaculos = criar_cenario(int(args[0]) if args else None)
    print("semente:", semente)
    for k, (c, r) in enumerate(obstaculos):
        print(f"obstaculo {k}: centro=({c[0]:.2f}, {c[1]:.2f}) raio={r:.2f}")

    mapa = criar_mapa(obstaculos)

    pf.verbose = "--quiet" not in sys.argv   # log das forças vai para o terminal (stdout)
    t0 = time.perf_counter()
    traj_pf, ok_pf = rodar_pf(obstaculos)
    t_pf = time.perf_counter() - t0

    t0 = time.perf_counter()
    traj_as, ok_as = rodar_astar(mapa)
    t_as = time.perf_counter() - t0

    print()
    print(f"{'':<20}{'Potential Fields':>18}{'A*':>12}")
    print(f"{'chegou ao objetivo':<20}{str(ok_pf):>18}{str(ok_as):>12}")
    print(f"{'comprimento':<20}{comprimento(traj_pf):>18.2f}{comprimento(traj_as):>12.2f}")
    print(f"{'pontos':<20}{len(traj_pf):>18d}{len(traj_as):>12d}")
    print(f"{'tempo (ms)':<20}{t_pf * 1e3:>18.1f}{t_as * 1e3:>12.1f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, traj, ok, titulo, cor in [
        (axs[0], traj_pf, ok_pf, "Potential Fields (local)", "tab:blue"),
        (axs[1], traj_as, ok_as, "A* (global)", "tab:orange"),
    ]:
        ax.imshow(mapa == OBSTACULO, origin="lower",
                  extent=(-res / 2, 10 + res / 2, -res / 2, 10 + res / 2),
                  cmap="Greys", alpha=0.5)
        for centro, raio in obstaculos:
            ax.add_patch(plt.Circle(centro, raio, fill=False, color="k", lw=0.8))
        ax.plot(traj[:, 0], traj[:, 1], ".-", color=cor, markersize=2,
                label=f"caminho (L={comprimento(traj):.2f})")
        ax.plot(*INICIO, "go", markersize=10, label="início")
        ax.plot(*OBJETIVO, "r*", markersize=15, label="objetivo")
        ax.set_aspect("equal")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_title(titulo + ("" if ok else " — FALHOU"))
        ax.legend(loc="upper left")
    fig.suptitle(f"semente = {semente}")
    plt.tight_layout()
    plt.savefig("compare_planners.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
