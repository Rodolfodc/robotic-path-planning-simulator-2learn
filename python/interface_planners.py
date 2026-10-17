"""
Interface interativa: A*, Potential Fields e PID no mesmo cenário.

    python interface_planners.py

  - Aba "Exibir": liga/desliga o que aparece no gráfico (algoritmos, caminho
    planejado, caminho percorrido com PID, obstáculos, grade, ...).
  - Demais abas: parâmetros de cada parte, todos com um valor padrão.
  - Caixa "aleatório" ao lado de cada parâmetro: a cada execução um novo valor
    é sorteado dentro da faixa mostrada e fica visível no campo. Desmarque a
    caixa para FIXAR aquele valor (e repetir o experimento).
  - Para rodar de novo depois de mudar um parâmetro: botão "Executar" ou Enter.

Reaproveita (sem alterar) os módulos existentes: potential_fields.py, astar.py,
pid.py, follow_path.py e compare_planners.py.
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import potential_fields as pf
from astar import a_estrela, OBSTACULO, euclidiana, manhattan, chebyshev
from pid import PID
from follow_path import reamostrar, angulo_normalizado, colidiu
from compare_planners import comprimento

# ---------------------------------------------------------------
# 1. Parâmetros: (chave, rótulo, padrão, tipo, faixa do sorteio)
#    faixa = (mín, máx) para números, ou lista de opções para escolhas.
# ---------------------------------------------------------------
SECOES = [
    ("Cenário", [
        ("semente",       "semente",             42,    int,   (0, 99999)),
        ("n_obstaculos",  "nº de obstáculos",    4,     int,   (2, 8)),
        ("raio_min",      "raio mínimo",         0.5,   float, (0.3, 0.8)),
        ("raio_max",      "raio máximo",         1.2,   float, (0.9, 1.5)),
        ("folga",         "folga entre obstáculos", 0.5, float, (0.2, 1.0)),
        ("inicio_x",      "início x",            0.0,   float, (0.0, 2.0)),
        ("inicio_y",      "início y",            0.0,   float, (0.0, 2.0)),
        ("objetivo_x",    "objetivo x",          10.0,  float, (8.0, 10.0)),
        ("objetivo_y",    "objetivo y",          10.0,  float, (8.0, 10.0)),
    ]),
    ("Potential Fields", [
        ("k_att",      "ganho atrativo (k_att)",   1.0,  float, (0.5, 3.0)),
        ("k_rep",      "ganho repulsivo (k_rep)",  5.0,  float, (1.0, 15.0)),
        ("rho_0",      "zona de influência (rho_0)", 2.0, float, (1.0, 3.0)),
        ("passo",      "passo",                    0.05, float, (0.02, 0.1)),
        ("tolerancia", "tolerância ao objetivo",   0.1,  float, (0.05, 0.2)),
        ("max_iter",   "máx. de iterações",        2000, int,   (500, 4000)),
    ]),
    ("A*", [
        ("res",       "tamanho da célula",  0.2,         float, [0.2, 0.25, 0.5]),
        ("margem",    "margem de segurança", 0.1,        float, (0.0, 0.3)),
        ("distancia", "métrica de distância", "euclidiana", str,
         ["euclidiana", "manhattan", "chebyshev"]),
    ]),
    ("Robô e PID", [
        ("v_max",         "velocidade máxima",       1.5,  float, (0.5, 2.5)),
        ("espaco_pontos", "espaço entre pontos-alvo", 0.5, float, (0.25, 1.0)),
        ("raio_aceite",   "raio de aceitação",        0.3, float, (0.15, 0.5)),
        ("massa",         "massa",                    1.0, float, (0.5, 2.0)),
        ("inercia",       "inércia de giro",          0.5, float, (0.2, 1.0)),
        ("kp_v",          "velocidade: Kp",           3.0, float, (1.0, 6.0)),
        ("ki_v",          "velocidade: Ki",           0.5, float, (0.0, 1.0)),
        ("kd_v",          "velocidade: Kd",           0.0, float, (0.0, 0.3)),
        ("kp_th",         "direção: Kp",              4.0, float, (1.0, 8.0)),
        ("ki_th",         "direção: Ki",              0.0, float, (0.0, 0.3)),
        ("kd_th",         "direção: Kd",              1.0, float, (0.0, 2.0)),
    ]),
]
PARAMETROS = {c: (r, p, t, f) for _, ps in SECOES for c, r, p, t, f in ps}

# Itens da aba "Exibir": (chave, rótulo, padrão)
EXIBIR = [
    ("pf",         "Potential Fields",                   True),
    ("astar",      "A*",                                 True),
    ("planejado",  "Caminho planejado (sem PID)",        True),
    ("percorrido", "Caminho percorrido (com PID)",       True),
    ("obstaculos", "Obstáculos",                         True),
    ("influencia", "Zona de influência (rho_0)",         False),
    ("grade",      "Grade bloqueada do A*",              False),
    ("pontos",     "Pontos-alvo do seguidor",            False),
]

CORES = {"pf": ("tab:blue", "navy"), "astar": ("tab:orange", "saddlebrown")}
NOMES = {"pf": "Potential Fields", "astar": "A*"}
METRICAS = {"euclidiana": euclidiana, "manhattan": manhattan, "chebyshev": chebyshev}

# Constantes da física do robô (não expostas na interface)
DT = 0.05
T_MAX = 80.0
ATRITO_LIN = 0.5
ATRITO_ANG = 0.5
FORCA_MAX = 3.0
TORQUE_MAX = 3.0
TOL_OBJETIVO = 0.2


def sortear(tipo, faixa, rng):
    """Sorteia um valor na faixa (ou entre as opções) do parâmetro."""
    if isinstance(faixa, list):
        return faixa[int(rng.integers(len(faixa)))]
    lo, hi = faixa
    if tipo is int:
        return int(rng.integers(lo, hi + 1))
    return round(float(rng.uniform(lo, hi)), 2)


# ---------------------------------------------------------------
# 2. Cálculo (sem interface gráfica; roda numa thread)
# ---------------------------------------------------------------
def planejar_pf(p, obstaculos, inicio, objetivo):
    pf.k_att, pf.k_rep, pf.rho_0 = p["k_att"], p["k_rep"], p["rho_0"]
    pf.passo, pf.tolerancia, pf.max_iter = p["passo"], p["tolerancia"], int(p["max_iter"])
    pf.verbose = False
    caminho, ok = pf.planejar(inicio, objetivo, obstaculos)
    # com ganhos ruins o método diverge (valores enormes/NaN): corta no ponto sensato
    valido = np.all(np.isfinite(caminho), axis=1) & np.all(np.abs(caminho) < 1e3, axis=1)
    if not valido.all():
        caminho, ok = caminho[:int(np.argmin(valido))], False
        if len(caminho) == 0:
            caminho = np.array([inicio])
    return caminho, ok


def criar_grade(p, obstaculos):
    """Rasteriza os círculos. mapa[y, x]: 1 = livre, OBSTACULO = bloqueada."""
    res, margem = p["res"], p["margem"]
    n = int(round(10 / res)) + 1
    xs = np.arange(n) * res
    X, Y = np.meshgrid(xs, xs)
    mapa = np.ones((n, n))
    for centro, raio in obstaculos:
        mapa[np.hypot(X - centro[0], Y - centro[1]) <= raio + margem] = OBSTACULO
    return mapa


def planejar_astar(p, mapa, inicio, objetivo):
    res = p["res"]
    cel_ini = tuple(int(v) for v in np.round(inicio / res))
    cel_obj = tuple(int(v) for v in np.round(objetivo / res))
    mapa = mapa.copy()
    mapa[cel_ini[1], cel_ini[0]] = 1       # início e objetivo nunca são bloqueados
    mapa[cel_obj[1], cel_obj[0]] = 1
    cam, _ = a_estrela(mapa, cel_ini, cel_obj, distancia=METRICAS[p["distancia"]])
    if cam is None:
        return np.array([inicio]), False
    cam = np.array(cam, dtype=float) * res
    cam[0], cam[-1] = inicio, objetivo     # extremos exatos (a célula é arredondada)
    return cam, True


def erro_rastreamento(trajetoria, caminho):
    """Distância de cada ponto percorrido até o caminho planejado.
    Em blocos e com o caminho limitado a ~2000 pontos, para a memória ficar sempre pequena."""
    denso = reamostrar(caminho, max(0.02, comprimento(caminho) / 2000))
    erros = np.empty(len(trajetoria))
    for i in range(0, len(trajetoria), 100):
        bloco = trajetoria[i:i + 100]
        d = np.linalg.norm(bloco[:, None, :] - denso[None, :, :], axis=-1)
        erros[i:i + 100] = d.min(axis=1)
    return erros


def seguir(caminho, p):
    """Robô com inércia seguindo o caminho com dois PIDs (velocidade e direção).
    Estado: (x, y), ângulo th, velocidade v, velocidade angular w."""
    if len(caminho) < 2:
        return caminho.copy(), False, caminho.copy()

    pontos = reamostrar(caminho, p["espaco_pontos"])
    pid_v = PID(p["kp_v"], p["ki_v"], p["kd_v"])
    pid_th = PID(p["kp_th"], p["ki_th"], p["kd_th"])

    x, y = caminho[0]
    th = np.arctan2(pontos[0][1] - y, pontos[0][0] - x)
    v = w = 0.0
    k = 0
    chegou = False
    trajetoria = [(x, y)]

    for _ in range(int(T_MAX / DT)):
        alvo = pontos[k]
        dist = np.hypot(alvo[0] - x, alvo[1] - y)

        ultimo = k == len(pontos) - 1
        if dist < (TOL_OBJETIVO if ultimo else p["raio_aceite"]):
            if ultimo:
                chegou = True
                break
            k += 1
            alvo = pontos[k]
            dist = np.hypot(alvo[0] - x, alvo[1] - y)

        erro_th = angulo_normalizado(np.arctan2(alvo[1] - y, alvo[0] - x) - th)
        tau = pid_th.calcular(erro_th, 0.0, DT)

        v_des = p["v_max"] * max(0.0, np.cos(erro_th)) ** 3
        if ultimo:
            v_des = min(v_des, 1.5 * dist)
        F = pid_v.calcular(v_des, v, DT)

        F = np.clip(F, -FORCA_MAX, FORCA_MAX)
        tau = np.clip(tau, -TORQUE_MAX, TORQUE_MAX)

        v += (F - ATRITO_LIN * v) / p["massa"] * DT
        w += (tau - ATRITO_ANG * w) / p["inercia"] * DT
        th += w * DT
        x += v * np.cos(th) * DT
        y += v * np.sin(th) * DT
        trajetoria.append((x, y))

    return np.array(trajetoria), chegou, pontos


def calcular(p, usar_pf, usar_astar, com_pid, com_grade):
    """Gera o cenário e roda o que foi pedido. Retorna um dicionário de resultados."""
    inicio = np.array([p["inicio_x"], p["inicio_y"]], dtype=float)
    objetivo = np.array([p["objetivo_x"], p["objetivo_y"]], dtype=float)

    # parâmetros que o gerador de obstáculos lê do módulo potential_fields
    pf.raio_min, pf.raio_max, pf.folga, pf.rho_0 = p["raio_min"], p["raio_max"], p["folga"], p["rho_0"]
    rng = np.random.default_rng(p["semente"])
    obstaculos = pf.gerar_obstaculos(p["n_obstaculos"], rng, inicio, objetivo)

    mapa = criar_grade(p, obstaculos) if (usar_astar or com_grade) else None

    planos = {}
    escolhidos = [("pf", usar_pf), ("astar", usar_astar)]
    for chave, ativo in escolhidos:
        if not ativo:
            continue
        t0 = time.perf_counter()
        if chave == "pf":
            caminho, ok = planejar_pf(p, obstaculos, inicio, objetivo)
        else:
            caminho, ok = planejar_astar(p, mapa, inicio, objetivo)
        info = {"caminho": caminho, "ok": ok, "t_plano": time.perf_counter() - t0,
                "L_plano": comprimento(caminho) if len(caminho) > 1 else 0.0}

        if com_pid:
            traj, chegou, pontos = seguir(caminho, p)
            info.update(trajetoria=traj, chegou=chegou, pontos=pontos,
                        L_real=comprimento(traj) if len(traj) > 1 else 0.0,
                        erro=erro_rastreamento(traj, caminho),
                        bateu=colidiu(traj, obstaculos),
                        t_sim=(len(traj) - 1) * DT)
        planos[chave] = info

    return {"params": p, "inicio": inicio, "objetivo": objetivo,
            "obstaculos": obstaculos, "mapa": mapa, "planos": planos}


# ---------------------------------------------------------------
# 3. Interface
# ---------------------------------------------------------------
class App:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("A*, Potential Fields e PID")
        raiz.protocol("WM_DELETE_WINDOW", self._fechar)

        self.vars = {}           # chave -> StringVar com o valor mostrado
        self.aleatorio = {}      # chave -> BooleanVar (sorteia a cada execução?)
        self.widgets = {}        # chave -> campo de entrada
        self.exibir = {}         # chave -> BooleanVar (mostrar no gráfico?)
        self.rng = np.random.default_rng()
        self.fila = queue.Queue()
        self.resultado = None
        self.ocupado = False

        self._montar_painel()
        self._montar_grafico()
        self.executar()

    # ---------- montagem ----------
    def _montar_painel(self):
        esquerda = ttk.Frame(self.raiz, padding=8)
        esquerda.pack(side="left", fill="y")

        botoes = ttk.Frame(esquerda)
        botoes.pack(fill="x", pady=(0, 8))
        self.botoes = [
            ttk.Button(botoes, text="Executar", command=self.executar),
            ttk.Button(botoes, text="Sortear tudo", command=self._sortear_tudo),
            ttk.Button(botoes, text="Fixar valores", command=self._fixar_valores),
            ttk.Button(botoes, text="Padrões", command=self._padroes),
        ]
        for b in self.botoes:
            b.pack(side="left", padx=2)

        self.abas = ttk.Notebook(esquerda)
        self.abas.pack(fill="y", expand=True)

        aba = ttk.Frame(self.abas, padding=8)
        self.abas.add(aba, text="Exibir")
        for i, (chave, rotulo, padrao) in enumerate(EXIBIR):
            self.exibir[chave] = tk.BooleanVar(value=padrao)
            ttk.Checkbutton(aba, text=rotulo, variable=self.exibir[chave],
                            command=self._ao_mudar_exibicao).grid(row=i, column=0, sticky="w", pady=2)

        for nome, parametros in SECOES:
            self._montar_aba(nome, parametros)

        self.status = tk.StringVar(value="")
        ttk.Label(esquerda, textvariable=self.status, wraplength=320).pack(anchor="w", pady=(8, 0))

    def _montar_aba(self, nome, parametros):
        aba = ttk.Frame(self.abas, padding=8)
        self.abas.add(aba, text=nome)
        for col, titulo in enumerate(["parâmetro", "valor", "aleatório", "faixa"]):
            ttk.Label(aba, text=titulo, foreground="gray").grid(row=0, column=col, sticky="w", padx=4)

        for i, (chave, rotulo, padrao, tipo, faixa) in enumerate(parametros, start=1):
            ttk.Label(aba, text=rotulo).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            self.vars[chave] = tk.StringVar(value=str(padrao))
            if isinstance(faixa, list):
                campo = ttk.Combobox(aba, textvariable=self.vars[chave], width=11, state="readonly",
                                     values=[str(o) for o in faixa])
                texto_faixa = "opções"
            else:
                campo = ttk.Entry(aba, textvariable=self.vars[chave], width=12)
                campo.bind("<Return>", lambda e: self.executar())
                texto_faixa = f"{faixa[0]} – {faixa[1]}"
            campo.grid(row=i, column=1, padx=4)
            self.widgets[chave] = campo

            self.aleatorio[chave] = tk.BooleanVar(value=False)
            ttk.Checkbutton(aba, variable=self.aleatorio[chave],
                            command=lambda c=chave: self._atualizar_estado(c)).grid(row=i, column=2)
            ttk.Label(aba, text=texto_faixa, foreground="gray").grid(row=i, column=3, sticky="w", padx=4)

    def _montar_grafico(self):
        direita = ttk.Frame(self.raiz, padding=8)
        direita.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=direita)
        barra = NavigationToolbar2Tk(self.canvas, direita, pack_toolbar=False)
        barra.pack(side="top", fill="x")

        self.texto = tk.Text(direita, height=7, font="TkFixedFont", state="disabled")
        self.texto.pack(side="bottom", fill="x", pady=(8, 0))
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # ---------- ações dos botões ----------
    def _atualizar_estado(self, chave):
        self.widgets[chave].state(["disabled"] if self.aleatorio[chave].get() else ["!disabled"])

    def _sortear_tudo(self):
        for chave in self.aleatorio:
            self.aleatorio[chave].set(True)
            self._atualizar_estado(chave)
        self.executar()

    def _fixar_valores(self):
        for chave in self.aleatorio:
            self.aleatorio[chave].set(False)
            self._atualizar_estado(chave)

    def _padroes(self):
        for chave, (_, padrao, _, _) in PARAMETROS.items():
            self.vars[chave].set(str(padrao))
            self.aleatorio[chave].set(False)
            self._atualizar_estado(chave)
        self.executar()

    def _ao_mudar_exibicao(self):
        """Só recalcula se o que foi ligado ainda não foi calculado."""
        r = self.resultado
        faltando = r is None
        if r is not None:
            for chave in ("pf", "astar"):
                if self.exibir[chave].get() and chave not in r["planos"]:
                    faltando = True
            if self.exibir["percorrido"].get():
                if any("trajetoria" not in info for info in r["planos"].values()):
                    faltando = True
            if self.exibir["grade"].get() and r["mapa"] is None:
                faltando = True
        if faltando:
            self.executar()
        else:
            self.redesenhar()

    # ---------- leitura dos parâmetros ----------
    def _ler_parametros(self):
        p = {}
        for chave, (rotulo, _, tipo, faixa) in PARAMETROS.items():
            if self.aleatorio[chave].get():
                valor = sortear(tipo, faixa, self.rng)
                self.vars[chave].set(str(valor))      # mostra o valor sorteado
            else:
                texto = self.vars[chave].get().strip().replace(",", ".")
                try:
                    valor = int(float(texto)) if tipo is int else tipo(texto)
                except ValueError:
                    raise ValueError(f"Valor inválido em '{rotulo}': {texto!r}")
            p[chave] = valor

        if p["raio_max"] < p["raio_min"]:
            p["raio_min"], p["raio_max"] = p["raio_max"], p["raio_min"]
            self.vars["raio_min"].set(str(p["raio_min"]))
            self.vars["raio_max"].set(str(p["raio_max"]))
        for chave in ("passo", "res", "max_iter", "v_max", "massa", "inercia", "espaco_pontos", "raio_aceite"):
            if p[chave] <= 0:
                raise ValueError(f"'{PARAMETROS[chave][0]}' precisa ser maior que zero.")
        for chave in ("inicio_x", "inicio_y", "objetivo_x", "objetivo_y"):
            if not 0 <= p[chave] <= 10:
                raise ValueError(f"'{PARAMETROS[chave][0]}' precisa estar entre 0 e 10.")
        return p

    # ---------- execução (thread) ----------
    def executar(self):
        if self.ocupado:
            return
        try:
            p = self._ler_parametros()
        except ValueError as e:
            self.status.set(str(e))
            return

        pedido = dict(usar_pf=self.exibir["pf"].get(), usar_astar=self.exibir["astar"].get(),
                      com_pid=self.exibir["percorrido"].get(), com_grade=self.exibir["grade"].get())
        self.ocupado = True
        for b in self.botoes:
            b.state(["disabled"])
        self.status.set("Calculando...")

        def trabalho():
            try:
                self.fila.put(("ok", calcular(p, **pedido)))
            except Exception as e:                     # mostra o erro na interface
                self.fila.put(("erro", f"{type(e).__name__}: {e}"))

        threading.Thread(target=trabalho, daemon=True).start()
        self.raiz.after(100, self._verificar)

    def _verificar(self):
        try:
            tipo, dado = self.fila.get_nowait()
        except queue.Empty:
            self.raiz.after(100, self._verificar)
            return
        self.ocupado = False
        for b in self.botoes:
            b.state(["!disabled"])
        if tipo == "ok":
            self.resultado = dado
            self.status.set(f"Pronto. semente = {dado['params']['semente']}")
            self.redesenhar()
        else:
            self.status.set(f"Erro: {dado}")

    # ---------- desenho ----------
    def redesenhar(self):
        r = self.resultado
        if r is None:
            return
        p, ax = r["params"], self.ax
        ax.clear()

        if self.exibir["grade"].get() and r["mapa"] is not None:
            m = p["res"] / 2
            ax.imshow(r["mapa"] == OBSTACULO, origin="lower", extent=(-m, 10 + m, -m, 10 + m),
                      cmap="Greys", alpha=0.45)
        if self.exibir["obstaculos"].get():
            for centro, raio in r["obstaculos"]:
                ax.add_patch(Circle(centro, raio, fill=True, color="gray", alpha=0.7))
        if self.exibir["influencia"].get():
            for centro, raio in r["obstaculos"]:
                ax.add_patch(Circle(centro, raio + p["rho_0"], fill=False, ls="--",
                                         color="gray", alpha=0.5))

        for chave, info in r["planos"].items():
            if not self.exibir[chave].get():
                continue
            claro, escuro = CORES[chave]
            nome = NOMES[chave] + ("" if info["ok"] else " (FALHOU)")
            if self.exibir["planejado"].get():
                c = info["caminho"]
                ax.plot(c[:, 0], c[:, 1], "--", color=claro, lw=1.6,
                        label=f"{nome}: planejado, sem PID (L={info['L_plano']:.2f})")
            if self.exibir["percorrido"].get() and "trajetoria" in info:
                t = info["trajetoria"]
                ax.plot(t[:, 0], t[:, 1], "-", color=escuro, lw=2.2,
                        label=f"{nome}: percorrido, com PID (L={info['L_real']:.2f})")
                if self.exibir["pontos"].get():
                    ax.plot(info["pontos"][:, 0], info["pontos"][:, 1], ".", color=escuro, markersize=5)

        ax.plot(*r["inicio"], "go", markersize=10, label="início")
        ax.plot(*r["objetivo"], "r*", markersize=15, label="objetivo")
        ax.set_aspect("equal")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.grid(alpha=0.2)
        ax.set_title(f"semente = {p['semente']}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._escrever_metricas()

    def _escrever_metricas(self):
        linhas = [f"{'':<18}{'plano ok':>9}{'L plan.':>9}{'L real':>8}{'erro máx':>10}"
                  f"{'chegou':>8}{'colisão':>9}{'plan. (ms)':>11}"]
        for chave, info in self.resultado["planos"].items():
            if not self.exibir[chave].get():
                continue
            linha = f"{NOMES[chave]:<18}{str(info['ok']):>9}{info['L_plano']:>9.2f}"
            if "trajetoria" in info:
                linha += (f"{info['L_real']:>8.2f}{info['erro'].max():>10.2f}{str(info['chegou']):>8}"
                          f"{('SIM' if info['bateu'] else 'não'):>9}")
            else:
                linha += f"{'-':>8}{'-':>10}{'-':>8}{'-':>9}"
            linha += f"{info['t_plano'] * 1e3:>11.1f}"
            linhas.append(linha)
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", "\n".join(linhas))
        self.texto.configure(state="disabled")

    def _fechar(self):
        self.raiz.quit()
        self.raiz.destroy()


def main():
    raiz = tk.Tk()
    raiz.geometry("1300x820")
    App(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
