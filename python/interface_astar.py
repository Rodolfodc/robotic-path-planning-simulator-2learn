# ------------------------------------------------------------------------
# Interface interativa para o A* (escolha de obstaculos, inicio e objetivo)
# ------------------------------------------------------------------------
# Executar no terminal (com astar.py na mesma pasta):
#     pip install numpy matplotlib
#     python interface_astar.py

import sys

import numpy as np
import matplotlib.pyplot as plt

from astar import a_estrela, OBSTACULO

# Numero de linhas e colunas da grade
NL = 8
NC = 8

# Mostra os valores de F e avanca passo a passo so em grades pequenas
TEXTO_NUM = NL < 11 and NC < 11


# ------------------------------------------------------------------------
# Utilitarios de janela
# ------------------------------------------------------------------------
def arredonda(v):
    # round do MATLAB (meio para longe do zero); o round do Python e bancario
    return int(np.floor(v + 0.5))


def encerra_se_fechada(fig):
    # Se o usuario fechar a janela, encerra o script em vez de travar o cmd
    if not plt.fignum_exists(fig.number):
        print('Janela fechada. Encerrando.')
        sys.exit(0)


def clique(fig):
    # Equivalente a [x, y, botao] = ginput(1) do MATLAB, retornando a celula.
    # Cliques fora da grade sao ignorados.
    while True:
        evento = {}

        def ao_clicar(e):
            if e.inaxes is not None:
                evento['x'], evento['y'], evento['b'] = e.xdata, e.ydata, int(e.button)

        cid = fig.canvas.mpl_connect('button_press_event', ao_clicar)
        while not evento:
            encerra_se_fechada(fig)
            plt.pause(0.05)
        fig.canvas.mpl_disconnect(cid)

        x, y = arredonda(evento['x']), arredonda(evento['y'])
        if 0 <= x < NC and 0 <= y < NL:
            return x, y, evento['b']


def aguarda_tecla(fig):
    # Equivalente ao "pause" sem argumento do MATLAB: espera uma tecla
    while True:
        encerra_se_fechada(fig)
        if plt.waitforbuttonpress():  # True = tecla, False = mouse
            return


def criar_interface(NL, NC):
    # Cria a janela com a grade NL x NC (celulas centradas em 0..NC-1, 0..NL-1)
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Algoritmo A*')
    for i in np.arange(-0.5, NC):
        ax.plot([i, i], [-0.5, NL - 0.5], 'k')
    for i in np.arange(-0.5, NL):
        ax.plot([-0.5, NC - 0.5], [i, i], 'k')
    ax.set_aspect('equal')
    plt.show(block=False)
    plt.pause(0.1)
    return fig, ax


# ------------------------------------------------------------------------
# Escolhas interativas
# ------------------------------------------------------------------------
def escolher_obstaculos(fig, ax, mapa):
    msg = 'Posicione os obstaculos com o botao esquerdo e ultimo com o botao direito.'
    ax.set_xlabel(msg)
    print(msg)
    botao = 1
    while botao == 1:  # botao esquerdo = 1, direito = 3
        x, y, botao = clique(fig)
        mapa[y, x] = OBSTACULO
        ax.plot(x, y, 'sk', markersize=20, linewidth=4)
        ax.plot(x, y, '*k', markersize=20, linewidth=4)
        fig.canvas.draw_idle()


def escolher_ponto(fig, ax, msg, cor, rotulo, dx_rotulo):
    ax.set_xlabel(msg)
    print(msg)
    x, y, _ = clique(fig)
    ax.plot(x, y, 'o' + cor)
    ax.plot(x, y, '*' + cor)
    ax.text(x + dx_rotulo, y + 0.3, rotulo, color={'b': 'blue', 'r': 'red'}[cor])
    return x, y


# ------------------------------------------------------------------------
# Visualizacao dos passos do A*
# ------------------------------------------------------------------------
def criar_visualizador(fig, ax):
    verde = np.array([60, 153, 95]) / 256

    def ao_evento(tipo, x, y, *dados):
        if tipo == 'aberta':
            (f,) = dados
            ax.plot(x, y, 'o', color=verde, linewidth=2)
            if TEXTO_NUM:
                ax.text(x - 0.25, y - 0.25, f'{f:.2f}')
            plt.pause(0.001)

        elif tipo == 'atualizada':
            f, f_antigo = dados
            if TEXTO_NUM:
                ax.text(x - 0.25, y - 0.25, f'{f_antigo:.2f}', color='white')  # "apaga"
                ax.plot(x, y, 'om', linewidth=2)
                ax.text(x - 0.25, y - 0.25, f'{f:.2f}')
            plt.pause(0.001)

        elif tipo == 'expandida':
            (abertas,) = dados
            if TEXTO_NUM:
                aguarda_tecla(fig)
            else:
                plt.pause(0.000001)
            ax.plot(x, y, '.m', linewidth=2, markersize=15)
            plt.pause(0.001)
            print('-------------------------------------')
            print(np.round(abertas).astype(int))

    return ao_evento


def desenhar_caminho(ax, caminho, custo):
    ax.set_title(f'Custo total do caminho: {custo:g}')
    ax.set_xlabel('Melhor caminho encontrado por A*.')
    for x, y in reversed(caminho):  # do objetivo ao inicio, como no original
        ax.plot(x, y, 'ob', linewidth=2, markersize=10)
        plt.pause(0.05)
    xs, ys = zip(*caminho)
    ax.plot(xs, ys, 'b', linewidth=2)


# ------------------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------------------
def main():
    mapa = np.ones((NL, NC))
    fig, ax = criar_interface(NL, NC)

    escolher_obstaculos(fig, ax, mapa)
    inicio = escolher_ponto(fig, ax, 'Posicione o robo no Ponto Inicial com o botao esquerdo.',
                            'b', 'Inicio', -0.25)
    objetivo = escolher_ponto(fig, ax, 'Escolha o local do Ponto Objetivo com o botao esquerdo do mouse.',
                              'r', 'Objetivo', -0.4)
    mapa[objetivo[1], objetivo[0]] = 1  # o objetivo nunca e obstaculo (peso livre)

    if TEXTO_NUM:
        ax.text(inicio[0] - 0.25, inicio[1] - 0.25, f'{0:.2f}')
        ax.set_xlabel('Aperte uma tecla para evoluir o algoritmo.')
    else:
        ax.set_xlabel('Analisando o caminho.')

    caminho, custo = a_estrela(mapa, inicio, objetivo, ao_evento=criar_visualizador(fig, ax))

    if caminho is None:
        print('Todas as celulas foram analisadas. Nao existe caminho.')
    else:
        desenhar_caminho(ax, caminho, custo)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
