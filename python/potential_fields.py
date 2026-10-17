"""
Potential Fields (Campos Potenciais) -- versão mínima e didática.

Ideia (Khatib, 1986): o robô é uma partícula num campo de energia potencial U(q).
  - O objetivo gera um potencial ATRATIVO (um "vale" no objetivo).
  - Cada obstáculo gera um potencial REPULSIVO (um "morro" ao redor dele).
O robô se move na direção da força F = -grad U, isto é, "desce a ladeira".

Uso em outros scripts:

    from potential_fields import gerar_obstaculos, planejar
    obstaculos = gerar_obstaculos(4, rng)               # [(centro, raio), ...]
    trajetoria, chegou = planejar(inicio, objetivo, obstaculos)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Parâmetros
# ---------------------------------------------------------------
k_att = 1.0    # ganho atrativo
k_rep = 5.0    # ganho repulsivo
rho_0 = 2.0    # distância de influência do obstáculo
passo = 0.05   # tamanho do passo (taxa da descida de gradiente)
tolerancia = 0.1
max_iter = 2000

# Obstáculos aleatórios
raio_min, raio_max = 0.5, 1.2
folga = 0.5    # distância mínima de início/objetivo/outros obstáculos

verbose = True  # imprime o passo a passo das forças


def _log(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# ---------------------------------------------------------------
# 2. Cenário
# ---------------------------------------------------------------
def gerar_obstaculos(n, rng, inicio=None, objetivo=None):
    """Sorteia até n obstáculos circulares (centro, raio) sem cobrir
    início/objetivo e sem sobreposição entre si."""
    inicio = np.array([0.0, 0.0]) if inicio is None else inicio
    objetivo = np.array([10.0, 10.0]) if objetivo is None else objetivo
    obs = []
    tentativas = 0
    while len(obs) < n and tentativas < 1000:
        tentativas += 1
        centro = rng.uniform(1.0, 9.0, size=2)
        raio = rng.uniform(raio_min, raio_max)
        # não pode cobrir início ou objetivo
        if np.linalg.norm(centro - inicio) < raio + rho_0 / 2 + folga:
            continue
        if np.linalg.norm(centro - objetivo) < raio + rho_0 / 2 + folga:
            continue
        # não pode sobrepor outros obstáculos
        if any(np.linalg.norm(centro - c) < raio + r + folga for c, r in obs):
            continue
        obs.append((centro, raio))
    return obs


# ---------------------------------------------------------------
# 3. Força atrativa
#    U_att(q) = 1/2 * k_att * ||q - q_obj||^2
#    F_att(q) = -grad U_att = -k_att * (q - q_obj)
#    -> vetor que aponta para o objetivo, proporcional à distância.
# ---------------------------------------------------------------
def forca_atrativa(q, objetivo):
    _log("\n\n")
    _log("=========== attractive_force ==========")
    _log("k_att: ", k_att)
    _log("-k_att: ", (-k_att))
    _log("q: ", q)
    _log("objetivo: ", objetivo)
    _log("=========== =============== ==========")
    return -k_att * (q - objetivo)


# ---------------------------------------------------------------
# 4. Força repulsiva (de UM obstáculo)
#    rho = distância da borda do obstáculo até o robô
#    U_rep(q) = 1/2 * k_rep * (1/rho - 1/rho_0)^2   se rho <= rho_0
#             = 0                                    se rho >  rho_0
#    F_rep(q) = k_rep * (1/rho - 1/rho_0) * (1/rho^2) * grad(rho)
#    grad(rho) = vetor unitário que sai do obstáculo e aponta para o robô.
#    -> longe: nenhuma força. Perto: força que cresce rápido (tende a infinito).
# ---------------------------------------------------------------
def forca_repulsiva(q, centro, raio):
    _log("\n\n")
    _log("=========== respulsive_force ==========")
    _log("centro: ", centro)
    _log("raio: ", raio)
    _log("q: ", q)

    diferenca = q - centro
    _log("[diferenca = q - centro]")
    _log("diferenca: ", diferenca)
    dist_centro = np.linalg.norm(diferenca)
    _log("dist_centro: ", dist_centro)
    _log("rho = dist_centro - raio")
    rho = dist_centro - raio            # distância até a borda
    _log("rho: ", rho)
    _log("rho_0: ", rho_0)

    _log("rho > rho_0 : ", (rho > rho_0))
    if rho > rho_0:
        _log("FORA da zona de influência")
        _log("Vai retornar: ", np.zeros(2))
        return np.zeros(2)              # fora da zona de influência

    _log("DENTRO da Zona de influencia")
    _log("PREV rho: ", rho)
    _log("1e-3: ", 1e-3)
    rho = max(rho, 1e-3)                # evita divisão por zero
    _log("NEW rho: ", rho)

    _log(f"{diferenca=}")
    _log(f"{dist_centro=}")
    direcao = diferenca / dist_centro   # grad(rho), vetor unitário
    _log(f"{direcao=}")

    _log(f"{k_rep * (1/rho - 1/rho_0) * (1/rho**2) * direcao=}")

    _log("=========== =============== ===========")
    return k_rep * (1/rho - 1/rho_0) * (1/rho**2) * direcao


# ---------------------------------------------------------------
# 5. Laço principal: descida de gradiente
#    q_{k+1} = q_k + passo * F_total(q_k)
# ---------------------------------------------------------------
def planejar(inicio, objetivo, obstaculos):
    """Retorna (trajetoria Nx2, chegou: bool)."""
    q = np.array(inicio, dtype=float)
    objetivo = np.asarray(objetivo, dtype=float)
    trajetoria = [q.copy()]
    chegou = False

    for i in range(max_iter):
        F = forca_atrativa(q, objetivo)
        for centro, raio in obstaculos:
            F += forca_repulsiva(q, centro, raio)

        q = q + passo * F
        trajetoria.append(q.copy())

        if np.linalg.norm(q - objetivo) < tolerancia:
            _log(f"Objetivo alcançado em {i + 1} iterações.")
            chegou = True
            break
    else:
        _log("Não chegou ao objetivo (provável mínimo local).")

    return np.array(trajetoria), chegou


# ---------------------------------------------------------------
# 6. Execução direta: cenário aleatório + visualização
# ---------------------------------------------------------------
if __name__ == "__main__":
    inicio = np.array([0.0, 0.0])
    objetivo = np.array([10.0, 10.0])

    n_obstaculos = 4
    semente = None     # defina um inteiro para reproduzir o mesmo cenário
    if semente is None:
        semente = int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(semente)
    print("semente:", semente)

    obstaculos = gerar_obstaculos(n_obstaculos, rng, inicio, objetivo)
    print("=========== obstaculos ===========")
    for k, (c, r) in enumerate(obstaculos):
        print(f"obstaculo {k}: centro=({c[0]:.2f}, {c[1]:.2f}) raio={r:.2f}")
    print("==================================")

    trajetoria, _ = planejar(inicio, objetivo, obstaculos)

    fig, ax = plt.subplots(figsize=(6, 6))
    for centro, raio in obstaculos:
        ax.add_patch(plt.Circle(centro, raio, color="gray"))
        ax.add_patch(plt.Circle(centro, raio + rho_0, fill=False, ls="--", color="gray"))
    ax.plot(trajetoria[:, 0], trajetoria[:, 1], "b.-", markersize=2, label="trajetória")
    ax.plot(*inicio, "go", markersize=10, label="início")
    ax.plot(*objetivo, "r*", markersize=15, label="objetivo")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Potential Fields")
    plt.savefig("potential_fields.png", dpi=120)
    plt.show()
