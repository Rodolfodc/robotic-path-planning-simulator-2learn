"""
Seguir o caminho com PID -- planejador (A* / Potential Fields) + controle.

Os planejadores só dizem POR ONDE ir (uma lista de pontos). Um robô de verdade
tem inércia e não "teletransporta" de ponto em ponto. Aqui um robô com massa
tenta seguir cada caminho usando dois PIDs:
  - PID de VELOCIDADE: ajusta a força do motor para atingir a velocidade desejada.
  - PID de DIREÇÃO:    ajusta o torque para apontar o robô para o próximo ponto.

Para cada planejador o gráfico mostra:
  - SEM PID: o caminho planejado (ideal, o robô seria uma partícula sem inércia);
  - COM PID: o caminho realmente percorrido pelo robô.

Uso: python follow_path.py [semente] [--quiet] [--sem-janela]
  --quiet      : silencia o log das forças do potential_fields
  --sem-janela : só salva follow_path.png, sem abrir a janela do Matplotlib
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import potential_fields as pf
from compare_planners import (INICIO, OBJETIVO, criar_cenario, criar_mapa,
                              rodar_pf, rodar_astar, comprimento)
from pid import PID
from astar import OBSTACULO

# ---------------------------------------------------------------
# 1. Parâmetros do robô e do seguidor
# ---------------------------------------------------------------
dt = 0.05
T_max = 80.0           # tempo máximo de simulação (s)

massa = 1.0            # kg
inercia = 0.5          # momento de inércia (kg.m^2)
atrito_lin = 0.5       # atrito linear (N por m/s)
atrito_ang = 0.5       # atrito angular (N.m por rad/s)
forca_max = 3.0        # limite do motor (N)
torque_max = 3.0       # limite de giro (N.m)

v_max = 1.5            # velocidade de cruzeiro (m/s)
espaco_pontos = 0.5    # distância entre os pontos-alvo (m)
raio_aceite = 0.3      # a que distância o ponto conta como "alcançado" (m)
tol_objetivo = 0.2


# ---------------------------------------------------------------
# 2. Utilidades
# ---------------------------------------------------------------
def reamostrar(caminho, espaco):
    """Pontos igualmente espaçados ao longo do caminho (sempre inclui o último)."""
    caminho = np.asarray(caminho)
    seg = np.linalg.norm(np.diff(caminho, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    alvos = np.append(np.arange(espaco, s[-1], espaco), s[-1])
    return np.column_stack([np.interp(alvos, s, caminho[:, 0]),
                            np.interp(alvos, s, caminho[:, 1])])


def angulo_normalizado(a):
    return (a + np.pi) % (2 * np.pi) - np.pi      # força o ângulo para [-pi, pi]


def erro_de_rastreamento(trajetoria, caminho):
    """Distância de cada ponto percorrido até o ponto mais próximo do caminho planejado."""
    denso = reamostrar(caminho, 0.02)
    d = np.linalg.norm(trajetoria[:, None, :] - denso[None, :, :], axis=-1)
    return d.min(axis=1)


def colidiu(trajetoria, obstaculos):
    return any(np.any(np.linalg.norm(trajetoria - c, axis=1) < r) for c, r in obstaculos)


# ---------------------------------------------------------------
# 3. Robô com inércia seguindo uma lista de pontos
#    Estado: posição (x, y), ângulo th, velocidade v, velocidade angular w
#      m * dv/dt = F - atrito_lin * v
#      J * dw/dt = tau - atrito_ang * w
# ---------------------------------------------------------------
def seguir(caminho):
    pontos = reamostrar(caminho, espaco_pontos)
    pid_v = PID(kp=3.0, ki=0.5, kd=0.0)      # velocidade
    pid_th = PID(kp=4.0, ki=0.0, kd=1.0)     # direção

    x, y = caminho[0]
    th = np.arctan2(pontos[0][1] - y, pontos[0][0] - x)   # já começa apontando pro 1º ponto
    v = w = 0.0
    k = 0                                                   # índice do ponto-alvo atual
    chegou = False                                          # alcançou o fim do caminho?
    trajetoria = [(x, y)]

    for _ in range(int(T_max / dt)):
        alvo = pontos[k]
        dist = np.hypot(alvo[0] - x, alvo[1] - y)

        # chegou ao ponto-alvo? vai para o próximo (o último exige chegar de verdade)
        ultimo = k == len(pontos) - 1
        if dist < (tol_objetivo if ultimo else raio_aceite):
            if ultimo:
                chegou = True
                break
            k += 1
            alvo = pontos[k]
            dist = np.hypot(alvo[0] - x, alvo[1] - y)

        # --- PID de direção: erro = ângulo até o alvo - ângulo do robô
        erro_th = angulo_normalizado(np.arctan2(alvo[1] - y, alvo[0] - x) - th)
        tau = pid_th.calcular(erro_th, 0.0, dt)

        # --- PID de velocidade: anda devagar se está virado para o lado errado
        #     ou perto do último ponto
        v_des = v_max * max(0.0, np.cos(erro_th)) ** 3
        if ultimo:
            v_des = min(v_des, 1.5 * dist)
        F = pid_v.calcular(v_des, v, dt)

        # --- limites físicos dos atuadores
        F = np.clip(F, -forca_max, forca_max)
        tau = np.clip(tau, -torque_max, torque_max)

        # --- física (Euler)
        v += (F - atrito_lin * v) / massa * dt
        w += (tau - atrito_ang * w) / inercia * dt
        th += w * dt
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        trajetoria.append((x, y))

    trajetoria = np.array(trajetoria)
    return trajetoria, chegou


# ---------------------------------------------------------------
# 4. Programa principal
# ---------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    semente, obstaculos = criar_cenario(int(args[0]) if args else None)
    print("semente:", semente)
    for k, (c, r) in enumerate(obstaculos):
        print(f"obstaculo {k}: centro=({c[0]:.2f}, {c[1]:.2f}) raio={r:.2f}")

    pf.verbose = "--quiet" not in sys.argv
    mapa = criar_mapa(obstaculos)
    planos = {
        "Potential Fields": rodar_pf(obstaculos),
        "A*": rodar_astar(mapa),
    }

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    print()
    print(f"{'':<18}{'fim cam.':>9}{'L plan.':>10}{'L real':>9}{'erro máx':>10}{'erro médio':>12}{'colisão':>9}")
    for ax, (nome, (caminho, ok_plano)) in zip(axs, planos.items()):
        trajetoria, chegou = seguir(caminho)
        erro = erro_de_rastreamento(trajetoria, caminho)
        bateu = colidiu(trajetoria, obstaculos)
        print(f"{nome:<18}{str(chegou):>9}{comprimento(caminho):>10.2f}"
              f"{comprimento(trajetoria):>9.2f}{erro.max():>10.2f}{erro.mean():>12.3f}"
              f"{('SIM' if bateu else 'não'):>9}")

        for centro, raio in obstaculos:
            ax.add_patch(plt.Circle(centro, raio, color="gray", alpha=0.6))
        ax.plot(caminho[:, 0], caminho[:, 1], "--", color="k", lw=1.5,
                label=f"sem PID: planejado (L={comprimento(caminho):.2f})")
        ax.plot(trajetoria[:, 0], trajetoria[:, 1], "-", color="tab:red", lw=2,
                label=f"com PID: percorrido (L={comprimento(trajetoria):.2f})")
        ax.plot(*INICIO, "go", markersize=10, label="início")
        ax.plot(*OBJETIVO, "r*", markersize=15, label="objetivo")
        ax.set_aspect("equal")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        titulo = nome
        if not ok_plano:
            titulo += " — planejador FALHOU"
        elif bateu:
            titulo += " — COLIDIU"
        ax.set_title(titulo)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"semente = {semente}")
    plt.tight_layout()
    plt.savefig("follow_path.png", dpi=120)
    if "--sem-janela" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
