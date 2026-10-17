"""Algoritmo A* em grade 2D, sem dependencia de interface grafica.

Uso em outros scripts:

    import numpy as np
    from astar import a_estrela, OBSTACULO, manhattan

    mapa = np.ones((8, 8))            # peso de travessia de cada celula
    mapa[3, 2:6] = OBSTACULO          # mapa[y, x] = mapa[linha, coluna]
    mapa[5, :] = 2.0                  # faixa de terreno mais caro
    caminho, custo = a_estrela(mapa, inicio=(0, 0), objetivo=(7, 7))
    caminho, custo = a_estrela(mapa, (0, 0), (7, 7), distancia=manhattan)
    # caminho = [(x, y), ...] do inicio ao objetivo; (None, None) se nao houver

Convencoes:
- Coordenadas (x, y) = (coluna, linha), base 0, e o mapa e indexado mapa[y, x].
- Cada celula guarda seu peso de travessia (>= 0). Celulas iguais a
  OBSTACULO (-1) ou infinitas (np.inf) sao bloqueadas.
- Conectividade-8.
- Custo do passo = distancia(atual, vizinha) * peso da vizinha.
- Heuristica     = distancia(vizinha, objetivo) * menor peso livre do mapa.
  Com uma metrica (desigualdade triangular valida), essa heuristica e
  consistente e o caminho retornado e otimo.
"""

import numpy as np

OBSTACULO = -1

# Colunas da tabela Abertas
PERTENCE, X_PAI, Y_PAI, X, Y, G, F = range(7)


# ------------------------------------------------------------------------
# Metricas de distancia (estrategias). Qualquer funcao d(a, b) que seja uma
# metrica serve. Nao use distancias que violem a desigualdade triangular
# (ex.: euclidiana ao quadrado): a heuristica superestima e perde o otimo.
# ------------------------------------------------------------------------
def euclidiana(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def a_estrela(mapa, inicio, objetivo, distancia=euclidiana, ao_evento=None):
    """Busca o caminho de menor custo de `inicio` a `objetivo` com A*.

    Parametros
    ----------
    mapa : array NL x NC
        Peso de travessia de cada celula; OBSTACULO ou np.inf = bloqueada.
    inicio, objetivo : tupla (x, y)
        Coordenadas em base 0.
    distancia : funcao d(a, b) -> float
        Metrica usada no custo do passo e na heuristica. Padrao: euclidiana.
    ao_evento : funcao opcional
        Chamada a cada passo, para quem quiser visualizar a busca:
            ao_evento('aberta', x, y, F)                celula nova em Abertas
            ao_evento('atualizada', x, y, F, F_antigo)  caminho melhor encontrado
            ao_evento('expandida', x, y, abertas)       nova celula atual
        `abertas` e uma copia da tabela Abertas (colunas PERTENCE..F).

    Retorna
    -------
    (caminho, custo) : (lista de (x, y), float) ou (None, None) se nao ha caminho.
    """
    mapa = np.asarray(mapa, dtype=float)
    NL, NC = mapa.shape
    xIni, yIni = inicio
    xObj, yObj = objetivo
    notificar = ao_evento or (lambda *args: None)

    bloqueada = (mapa == OBSTACULO) | np.isinf(mapa)
    if np.any(mapa[~bloqueada] < 0):
        raise ValueError('Pesos negativos invalidam o A*. Use OBSTACULO ou np.inf para bloqueio.')

    # Limite inferior do peso: a heuristica supoe o terreno mais barato ate o fim
    livres = mapa[~bloqueada]
    peso_min = float(livres.min()) if livres.size else 0.0

    def h(x, y):
        return distancia((x, y), (xObj, yObj)) * peso_min

    # Celula inicial: g = 0 e F = 0
    xAtual, yAtual = xIni, yIni
    CustoIni2Atual = 0.0

    # Abertas: [ Pertence xPai yPai x y g F ]
    # Pertence = 1: ainda a explorar; 0: ja expandida
    Abertas = np.array([[0, xAtual, yAtual, xAtual, yAtual, 0.0, 0.0]])

    # Fechadas: celulas bloqueadas + celulas ja expandidas, como [x, y]
    yObst, xObst = np.nonzero(bloqueada)
    Fechadas = np.vstack([np.column_stack([xObst, yObst]), [xAtual, yAtual]])

    # ------------------------------------------------------------
    # Fase de ida: expande celulas ate chegar ao objetivo
    # ------------------------------------------------------------
    while xAtual != xObj or yAtual != yObj:

        for dx in (-1, 0, 1):      # eixo X (colunas)
            for dy in (-1, 0, 1):  # eixo Y (linhas)

                xViz = xAtual + dx
                yViz = yAtual + dy

                # Dentro do mapa e fora da lista de fechadas
                if not (0 <= xViz < NC and 0 <= yViz < NL):
                    continue
                if np.any((Fechadas[:, 0] == xViz) & (Fechadas[:, 1] == yViz)):
                    continue

                passo = distancia((xAtual, yAtual), (xViz, yViz)) * mapa[yViz, xViz]
                g = CustoIni2Atual + passo
                f = g + h(xViz, yViz)

                naLista = (Abertas[:, X] == xViz) & (Abertas[:, Y] == yViz)
                if np.sum(naLista) == 1:
                    # Ja esta em Abertas: atualiza se o novo F for menor
                    indAb = np.flatnonzero(naLista)[0]
                    if Abertas[indAb, F] > f:
                        F_antigo = Abertas[indAb, F]
                        Abertas[indAb, [X_PAI, Y_PAI, G, F]] = [xAtual, yAtual, g, f]
                        notificar('atualizada', xViz, yViz, f, F_antigo)
                else:
                    # Nova possibilidade a explorar
                    Abertas = np.vstack([Abertas, [1, xAtual, yAtual, xViz, yViz, g, f]])
                    notificar('aberta', xViz, yViz, f)

        # Celulas de Abertas ainda nao exploradas
        Livres = np.flatnonzero(Abertas[:, PERTENCE] == 1)
        if Livres.size == 0:
            return None, None  # Todas as celulas analisadas: nao existe caminho

        # Menor F; em empate, o primeiro da lista
        IndMinF = Livres[np.argmin(Abertas[Livres, F])]
        xAtual = int(Abertas[IndMinF, X])
        yAtual = int(Abertas[IndMinF, Y])
        CustoIni2Atual = Abertas[IndMinF, G]
        Fechadas = np.vstack([Fechadas, [xAtual, yAtual]])
        Abertas[IndMinF, PERTENCE] = 0

        notificar('expandida', xAtual, yAtual, Abertas.copy())

    # ------------------------------------------------------------
    # Fase de volta: do objetivo ao inicio seguindo os pais
    # ------------------------------------------------------------
    def ultimo_expandido(x, y):
        idx = np.flatnonzero((Abertas[:, X] == x) & (Abertas[:, Y] == y)
                             & (Abertas[:, PERTENCE] == 0))
        return idx[-1]

    custo = float(Abertas[ultimo_expandido(xObj, yObj), G])
    caminho = [(xObj, yObj)]
    x, y = xObj, yObj
    while x != xIni or y != yIni:
        ind = ultimo_expandido(x, y)
        x, y = int(Abertas[ind, X_PAI]), int(Abertas[ind, Y_PAI])
        caminho.append((x, y))
    caminho.reverse()

    return caminho, custo
