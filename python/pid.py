"""
PID (Proporcional-Integral-Derivativo) -- versão mínima e didática.

Ideia: o controlador olha o ERRO (e = alvo - medida) e calcula um comando u:
  - P (proporcional): reage ao erro de AGORA.     u_P = kp * e
  - I (integral):     reage ao erro ACUMULADO.    u_I = ki * soma(e * dt)
  - D (derivativo):   reage à TENDÊNCIA do erro.  u_D = kd * (de/dt)
  u = u_P + u_I + u_D

Exemplo: controle de velocidade de um carro (cruise control).
  - O motor aplica uma força u.
  - O atrito do ar freia o carro proporcional à velocidade.
  - Aos 10 s o carro pega uma SUBIDA (perturbação) que o freia ainda mais.
Compara P, PI e PID para ver o que cada termo faz.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Cenário
# ---------------------------------------------------------------
alvo = 10.0          # velocidade desejada (m/s)
v0 = 0.0             # velocidade inicial (m/s)
dt = 0.05            # passo de tempo (s)
T = 20.0             # duração da simulação (s)

massa = 1.0          # kg
atrito = 0.5         # coef. de atrito do ar (N por m/s)
t_subida = 10.0      # instante em que começa a subida (s)
forca_subida = 2.0   # força extra contra o carro na subida (N)


# ---------------------------------------------------------------
# 2. Controlador PID
#    Guarda a "memória" necessária: a integral e o erro anterior.
# ---------------------------------------------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.erro_anterior = None

    def calcular(self, alvo, medida, dt):
        erro = alvo - medida

        # P: proporcional ao erro atual
        p = self.kp * erro

        # I: soma do erro ao longo do tempo (elimina o erro em regime)
        self.integral += erro * dt
        i = self.ki * self.integral

        # D: taxa de variação do erro (amortece, evita ultrapassar o alvo)
        if self.erro_anterior is None:
            d = 0.0
        else:
            d = self.kd * (erro - self.erro_anterior) / dt
        self.erro_anterior = erro

        return p + i + d


# ---------------------------------------------------------------
# 3. Simulação do carro: m * dv/dt = u - atrito * v - perturbação
#    (integração de Euler: v_{k+1} = v_k + dv/dt * dt)
# ---------------------------------------------------------------
def simular(pid):
    t = np.arange(0.0, T, dt)
    v = v0
    velocidades, comandos = [], []

    for tk in t:
        u = pid.calcular(alvo, v, dt)                       # controlador decide
        perturbacao = forca_subida if tk >= t_subida else 0.0
        aceleracao = (u - atrito * v - perturbacao) / massa  # física do carro
        v = v + aceleracao * dt
        velocidades.append(v)
        comandos.append(u)

    return t, np.array(velocidades), np.array(comandos)


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # 4. Comparação: só P, PI e PID
    #    Obs.: com este passo de tempo (dt) e esta massa, um kd >= 1 deixa a
    #    simulação instável (o termo D amplifica o ruído numérico do Euler).
    # ---------------------------------------------------------------
    controladores = {
        "P   (kp=2)":              PID(kp=2.0, ki=0.0, kd=0.0),
        "PI  (kp=2, ki=1)":        PID(kp=2.0, ki=1.0, kd=0.0),
        "PID (kp=2, ki=1, kd=0.3)": PID(kp=2.0, ki=1.0, kd=0.3),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for nome, pid in controladores.items():
        t, v, u = simular(pid)
        erro_final = alvo - v[-1]
        print(f"{nome:<24} velocidade final = {v[-1]:.2f}  erro final = {erro_final:.2f}")
        ax1.plot(t, v, label=nome)
        ax2.plot(t, u, label=nome)

    ax1.axhline(alvo, color="k", ls="--", label="alvo")
    ax1.axvline(t_subida, color="gray", ls=":")
    ax1.text(t_subida + 0.2, 1, "subida", color="gray")
    ax1.set_ylabel("velocidade (m/s)")
    ax1.legend()
    ax1.set_title("PID: controle de velocidade")

    ax2.axvline(t_subida, color="gray", ls=":")
    ax2.set_ylabel("comando u (N)")
    ax2.set_xlabel("tempo (s)")

    plt.tight_layout()
    plt.savefig("pid.png", dpi=120)
    plt.show()
