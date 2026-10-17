/* Motor dos planejadores: Potential Fields, A* e seguidor com PID.
   Porta em JavaScript de potential_fields.py, astar.py, pid.py e follow_path.py.
   Não depende do navegador: roda também no Node (veja o fim do arquivo). */
const Engine = (() => {
  // ---------------------------------------------------------------
  // 1. Parâmetros: [chave, rótulo, padrão, tipo, faixa do sorteio, dica]
  //    faixa = [mín, máx] para números, ou lista de opções para escolhas.
  // ---------------------------------------------------------------
  const SECOES = [
    ["Cenário", [
      ["semente",      "Semente",                2,     "int",   [0, 99999], "Número que define o sorteio dos obstáculos. A mesma semente gera sempre o mesmo cenário."],
      ["n_obstaculos", "Nº de obstáculos",       4,     "int",   [2, 8],     "Quantos obstáculos circulares tentar colocar. Se não couberem com a folga pedida, aparecem menos."],
      ["raio_min",     "Raio mínimo",            0.5,   "float", [0.3, 0.8], "Menor raio que um obstáculo pode ter (o mapa vai de 0 a 10)."],
      ["raio_max",     "Raio máximo",            1.2,   "float", [0.9, 1.5], "Maior raio que um obstáculo pode ter. Se ficar menor que o mínimo, os dois são trocados."],
      ["folga",        "Folga entre obstáculos", 0.5,   "float", [0.2, 1.0], "Distância mínima entre obstáculos, e deles até o início e o objetivo. Evita cenários sem passagem."],
      ["inicio_x",     "Início x",               0,     "float", [0, 2],     "Posição horizontal de onde o robô parte (0 a 10)."],
      ["inicio_y",     "Início y",               0,     "float", [0, 2],     "Posição vertical de onde o robô parte (0 a 10)."],
      ["objetivo_x",   "Objetivo x",             10,    "float", [8, 10],    "Posição horizontal do destino (0 a 10)."],
      ["objetivo_y",   "Objetivo y",             10,    "float", [8, 10],    "Posição vertical do destino (0 a 10)."],
    ], { id: "cenario", chip: "Cenário", nota: "Vale para os dois algoritmos: os dois enfrentam os mesmos obstáculos." }],
    ["Potential Fields", [
      ["k_att",      "Ganho atrativo (k_att)",     1.0,  "float", [0.5, 3],    "Força com que o objetivo puxa o robô. Maior: vai mais direto ao destino, mas passa mais perto dos obstáculos."],
      ["k_rep",      "Ganho repulsivo (k_rep)",    5.0,  "float", [1, 15],     "Força com que os obstáculos empurram o robô. Maior: mantém mais distância, mas pode travar em passagens estreitas."],
      ["rho_0",      "Zona de influência (rho_0)", 2.0,  "float", [1, 3],      "Distância da borda do obstáculo até onde ele deixa de repelir. Fora dessa zona a força é zero. Também afasta os obstáculos do início e do objetivo ao sortear o cenário."],
      ["passo",      "Passo",                      0.05, "float", [0.02, 0.1], "Tamanho de cada passo da descida de gradiente. Pequeno: suave e lento. Grande: rápido, mas pode oscilar ou divergir."],
      ["tolerancia", "Tolerância ao objetivo",     0.1,  "float", [0.05, 0.2], "A que distância do objetivo o planejador considera que chegou."],
      ["max_iter",   "Máx. de iterações",          2000, "int",   [500, 4000], "Limite de passos. Se acabar antes de chegar, o planejador falha (provável mínimo local)."],
    ], { id: "pf", chip: "PF", nota: "Só afeta o Potential Fields. O A* ignora estes parâmetros." }],
    ["A*", [
      ["res",       "Tamanho da célula",    0.2,         "float", [0.1, 0.2, 0.25, 0.5], "Lado de cada célula da grade. Menor: caminho mais preciso, mas mais células para buscar."],
      ["margem",    "Margem de segurança",  0.1,         "float", [0, 0.3],              "Folga extra ao redor dos obstáculos ao marcar células como bloqueadas. Maior: caminho mais afastado."],
      ["distancia", "Métrica de distância", "euclidiana", "str",  ["euclidiana", "manhattan", "chebyshev"], "Como medir o custo de um passo e a distância até o objetivo. Euclidiana: em linha reta. Manhattan: soma dos eixos (diagonal custa 2). Chebyshev: o maior eixo (diagonal custa 1)."],
    ], { id: "astar", chip: "A*", nota: "Só afeta o A*. O Potential Fields ignora estes parâmetros." }],
    ["Robô e PID", [
      ["v_max",         "Velocidade máxima",        1.5, "float", [0.5, 2.5],  "Velocidade de cruzeiro que o PID de velocidade tenta atingir em linha reta. Muito alta, o robô passa dos pontos e corta curvas."],
      ["espaco_pontos", "Espaço entre pontos-alvo", 0.5, "float", [0.25, 1],   "O caminho planejado é dividido em pontos com esta distância entre si, e o robô vai de um a outro."],
      ["raio_aceite",   "Raio de aceitação",        0.3, "float", [0.15, 0.5], "A que distância de um ponto-alvo o robô o considera alcançado e passa ao próximo. Maior: corta mais as curvas."],
      ["massa",         "Massa",                    1.0, "float", [0.5, 2],    "Massa do robô. Maior: mais inércia, acelera e freia mais devagar."],
      ["inercia",       "Inércia de giro",          0.5, "float", [0.2, 1],    "Resistência a girar. Maior: o robô demora mais para virar e para parar de virar."],
      ["kp_v",          "Velocidade: Kp",           3.0, "float", [1, 6],      "Proporcional da velocidade: reage ao erro de agora entre a velocidade desejada e a real."],
      ["ki_v",          "Velocidade: Ki",           0.5, "float", [0, 1],      "Integral da velocidade: acumula o erro ao longo do tempo e elimina a diferença que sobra."],
      ["kd_v",          "Velocidade: Kd",           0.0, "float", [0, 0.3],    "Derivativo da velocidade: amortece variações rápidas. Valores altos deixam a simulação instável."],
      ["kp_th",         "Direção: Kp",              4.0, "float", [1, 8],      "Proporcional da direção: quanto o robô gira para cada radiano de erro entre sua direção e a do próximo ponto."],
      ["ki_th",         "Direção: Ki",              0.0, "float", [0, 0.3],    "Integral da direção: corrige um erro de ângulo que persiste. Normalmente fica em 0."],
      ["kd_th",         "Direção: Kd",              1.0, "float", [0, 2],      "Derivativo da direção: amortece o giro e evita oscilar em torno da direção certa."],
    ], { id: "pid", chip: "PID", nota: "Controla o robô que segue o caminho, tanto o do A* quanto o do Potential Fields. Não muda o caminho planejado, só o percorrido." }],
  ];

  const PARAMETROS = {};
  for (const [, ps, meta] of SECOES)
    for (const [c, r, d, t, f, dica] of ps) PARAMETROS[c] = { rotulo: r, padrao: d, tipo: t, faixa: f, dica, secao: meta.id };

  // Constantes da física do robô (não expostas)
  const DT = 0.05, T_MAX = 80, ATRITO_LIN = 0.5, ATRITO_ANG = 0.5;
  const FORCA_MAX = 3, TORQUE_MAX = 3, TOL_OBJETIVO = 0.2;

  // ---------------------------------------------------------------
  // 2. Utilidades
  // ---------------------------------------------------------------
  function criarRng(semente) {                       // mulberry32: rápido e reprodutível
    let a = (semente >>> 0) + 0x6D2B79F5;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sortear(chave, rnd = Math.random) {
    const { tipo, faixa } = PARAMETROS[chave];
    if (faixa.length > 2 || tipo === "str") return faixa[Math.floor(rnd() * faixa.length)];
    const [lo, hi] = faixa;
    if (tipo === "int") return lo + Math.floor(rnd() * (hi - lo + 1));
    return Math.round((lo + rnd() * (hi - lo)) * 100) / 100;
  }

  function validar(p) {
    for (const c of ["passo", "res", "max_iter", "v_max", "massa", "inercia", "espaco_pontos", "raio_aceite"])
      if (!(p[c] > 0)) return `"${PARAMETROS[c].rotulo}" precisa ser maior que zero.`;
    for (const c of ["inicio_x", "inicio_y", "objetivo_x", "objetivo_y"])
      if (!(p[c] >= 0 && p[c] <= 10)) return `"${PARAMETROS[c].rotulo}" precisa estar entre 0 e 10.`;
    for (const [c, { tipo }] of Object.entries(PARAMETROS))
      if (tipo !== "str" && !Number.isFinite(p[c])) return `Valor inválido em "${PARAMETROS[c].rotulo}".`;
    return null;
  }

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function comprimento(c) { let s = 0; for (let i = 1; i < c.length; i++) s += dist(c[i - 1], c[i]); return s; }
  const angulo = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

  function reamostrar(c, espaco) {                   // pontos igualmente espaçados
    const s = [0];
    for (let i = 1; i < c.length; i++) s.push(s[i - 1] + dist(c[i - 1], c[i]));
    const total = s[s.length - 1];
    const alvos = [];
    for (let a = espaco; a < total; a += espaco) alvos.push(a);
    alvos.push(total);
    const out = [];
    let j = 0;
    for (const a of alvos) {
      while (j < s.length - 2 && s[j + 1] < a) j++;
      const seg = s[j + 1] - s[j];
      const f = seg > 0 ? (a - s[j]) / seg : 0;
      out.push([c[j][0] + f * (c[j + 1][0] - c[j][0]), c[j][1] + f * (c[j + 1][1] - c[j][1])]);
    }
    return out;
  }

  function distAteSegmento(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    let t = l2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
  }

  function erroRastreamento(traj, caminho) {         // distância de cada ponto ao caminho planejado
    const erros = new Float64Array(traj.length);
    for (let i = 0; i < traj.length; i++) {
      let m = Infinity;
      for (let j = 1; j < caminho.length; j++) m = Math.min(m, distAteSegmento(traj[i], caminho[j - 1], caminho[j]));
      erros[i] = caminho.length > 1 ? m : dist(traj[i], caminho[0]);
    }
    return erros;
  }

  const colidiu = (traj, obs) => traj.some((q) => obs.some((o) => dist(q, o.c) < o.r));

  // ---------------------------------------------------------------
  // 3. Cenário
  // ---------------------------------------------------------------
  function gerarObstaculos(p, rnd, ini, obj) {
    const obs = [];
    let tentativas = 0;
    while (obs.length < p.n_obstaculos && tentativas < 1000) {
      tentativas++;
      const c = [1 + rnd() * 8, 1 + rnd() * 8];
      const r = p.raio_min + rnd() * (p.raio_max - p.raio_min);
      if (dist(c, ini) < r + p.rho_0 / 2 + p.folga) continue;
      if (dist(c, obj) < r + p.rho_0 / 2 + p.folga) continue;
      if (obs.some((o) => dist(c, o.c) < r + o.r + p.folga)) continue;
      obs.push({ c, r });
    }
    return obs;
  }

  // ---------------------------------------------------------------
  // 4. Potential Fields:  q_{k+1} = q_k + passo * F_total(q_k)
  //    F_att = -k_att (q - obj);  F_rep = k_rep (1/rho - 1/rho_0) / rho^2 * direção
  // ---------------------------------------------------------------
  function planejarPF(p, obs, ini, obj) {
    let x = ini[0], y = ini[1];
    const traj = [[x, y]];
    let ok = false;
    for (let i = 0; i < p.max_iter; i++) {
      let fx = -p.k_att * (x - obj[0]), fy = -p.k_att * (y - obj[1]);
      for (const { c, r } of obs) {
        const dx = x - c[0], dy = y - c[1], d = Math.hypot(dx, dy);
        let rho = d - r;
        if (rho > p.rho_0 || d < 1e-9) continue;
        rho = Math.max(rho, 1e-3);
        const mod = p.k_rep * (1 / rho - 1 / p.rho_0) / (rho * rho);
        fx += mod * dx / d; fy += mod * dy / d;
      }
      x += p.passo * fx; y += p.passo * fy;
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e3 || Math.abs(y) > 1e3) break;  // divergiu
      traj.push([x, y]);
      if (Math.hypot(x - obj[0], y - obj[1]) < p.tolerancia) { ok = true; break; }
    }
    return { caminho: traj, ok };
  }

  // ---------------------------------------------------------------
  // 5. A* em grade 8-conectada (mesma lógica do astar.py)
  // ---------------------------------------------------------------
  const METRICAS = {
    euclidiana: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]),
    manhattan: (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]),
    chebyshev: (a, b) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])),
  };

  function criarGrade(p, obs) {
    const n = Math.round(10 / p.res) + 1;
    const bloq = new Uint8Array(n * n);                // índice = y * n + x
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const q = [x * p.res, y * p.res];
      if (obs.some((o) => dist(q, o.c) <= o.r + p.margem)) bloq[y * n + x] = 1;
    }
    return { n, bloq };
  }

  class Heap {                                         // fila de prioridade mínima por f
    constructor() { this.a = []; }
    get size() { return this.a.length; }
    push(item) {
      const a = this.a; a.push(item);
      let i = a.length - 1;
      while (i > 0) { const pai = (i - 1) >> 1; if (a[pai][0] <= a[i][0]) break; [a[pai], a[i]] = [a[i], a[pai]]; i = pai; }
    }
    pop() {
      const a = this.a, topo = a[0], fim = a.pop();
      if (a.length) {
        a[0] = fim;
        let i = 0;
        for (;;) {
          let m = i; const l = 2 * i + 1, r = l + 1;
          if (l < a.length && a[l][0] < a[m][0]) m = l;
          if (r < a.length && a[r][0] < a[m][0]) m = r;
          if (m === i) break;
          [a[m], a[i]] = [a[i], a[m]]; i = m;
        }
      }
      return topo;
    }
  }

  function planejarAStar(p, grade, ini, obj) {
    const { n } = grade, bloq = grade.bloq.slice(), res = p.res;
    const d = METRICAS[p.distancia];
    const s = [Math.round(ini[0] / res), Math.round(ini[1] / res)];
    const g = [Math.round(obj[0] / res), Math.round(obj[1] / res)];
    bloq[s[1] * n + s[0]] = 0; bloq[g[1] * n + g[0]] = 0;   // início e objetivo nunca são bloqueados
    const gi = g[1] * n + g[0];
    const custo = new Float64Array(n * n).fill(Infinity);
    const pai = new Int32Array(n * n).fill(-1);
    const fechado = new Uint8Array(n * n);
    const aberta = new Heap();
    custo[s[1] * n + s[0]] = 0;
    aberta.push([d(s, g), s[1] * n + s[0]]);
    while (aberta.size) {
      const [, idx] = aberta.pop();
      if (fechado[idx]) continue;
      fechado[idx] = 1;
      if (idx === gi) {
        const cam = [];
        for (let k = idx; k !== -1; k = pai[k]) cam.push([(k % n) * res, Math.floor(k / n) * res]);
        cam.reverse();
        cam[0] = [ini[0], ini[1]]; cam[cam.length - 1] = [obj[0], obj[1]];   // extremos exatos
        return { caminho: cam, ok: true };
      }
      const x = idx % n, y = Math.floor(idx / n);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        if (!dx && !dy) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
        const ni = ny * n + nx;
        if (bloq[ni] || fechado[ni]) continue;
        const gn = custo[idx] + d([x, y], [nx, ny]);
        if (gn < custo[ni]) { custo[ni] = gn; pai[ni] = idx; aberta.push([gn + d([nx, ny], g), ni]); }
      }
    }
    return { caminho: [[ini[0], ini[1]]], ok: false };
  }

  // ---------------------------------------------------------------
  // 6. PID e robô com inércia
  //    m dv/dt = F - atrito v ;  J dw/dt = tau - atrito w
  // ---------------------------------------------------------------
  class PID {
    constructor(kp, ki, kd) { this.kp = kp; this.ki = ki; this.kd = kd; this.integral = 0; this.anterior = null; }
    calcular(alvo, medida, dt) {
      const erro = alvo - medida;
      this.integral += erro * dt;
      const d = this.anterior === null ? 0 : this.kd * (erro - this.anterior) / dt;
      this.anterior = erro;
      return this.kp * erro + this.ki * this.integral + d;
    }
  }

  function seguir(caminho, p) {
    if (caminho.length < 2) return { traj: caminho.map((q) => [...q]), chegou: false, pontos: caminho };
    const pontos = reamostrar(caminho, p.espaco_pontos);
    const pidV = new PID(p.kp_v, p.ki_v, p.kd_v), pidT = new PID(p.kp_th, p.ki_th, p.kd_th);
    let [x, y] = caminho[0];
    let th = Math.atan2(pontos[0][1] - y, pontos[0][0] - x), v = 0, w = 0, k = 0, chegou = false;
    const traj = [[x, y]];
    for (let i = 0; i < Math.floor(T_MAX / DT); i++) {
      let alvo = pontos[k], d = Math.hypot(alvo[0] - x, alvo[1] - y);
      const ultimo = k === pontos.length - 1;
      if (d < (ultimo ? TOL_OBJETIVO : p.raio_aceite)) {
        if (ultimo) { chegou = true; break; }
        k++; alvo = pontos[k]; d = Math.hypot(alvo[0] - x, alvo[1] - y);
      }
      const erroT = angulo(Math.atan2(alvo[1] - y, alvo[0] - x) - th);
      let tau = pidT.calcular(erroT, 0, DT);
      let vDes = p.v_max * Math.max(0, Math.cos(erroT)) ** 3;
      if (k === pontos.length - 1) vDes = Math.min(vDes, 1.5 * d);
      let F = pidV.calcular(vDes, v, DT);
      F = Math.max(-FORCA_MAX, Math.min(FORCA_MAX, F));
      tau = Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, tau));
      v += (F - ATRITO_LIN * v) / p.massa * DT;
      w += (tau - ATRITO_ANG * w) / p.inercia * DT;
      th += w * DT;
      x += v * Math.cos(th) * DT; y += v * Math.sin(th) * DT;
      traj.push([x, y]);
    }
    return { traj, chegou, pontos };
  }

  // ---------------------------------------------------------------
  // 7. Execução completa
  // ---------------------------------------------------------------
  function calcular(p) {
    const ini = [p.inicio_x, p.inicio_y], obj = [p.objetivo_x, p.objetivo_y];
    const obstaculos = gerarObstaculos(p, criarRng(p.semente), ini, obj);
    const grade = criarGrade(p, obstaculos);
    const planos = {};
    for (const chave of ["pf", "astar"]) {
      const t0 = performance.now();
      const { caminho, ok } = chave === "pf" ? planejarPF(p, obstaculos, ini, obj) : planejarAStar(p, grade, ini, obj);
      const tPlano = performance.now() - t0;
      const { traj, chegou, pontos } = seguir(caminho, p);
      const erro = erroRastreamento(traj, caminho);
      planos[chave] = {
        caminho, ok, tPlano, lPlano: comprimento(caminho),
        traj, chegou, pontos, lReal: comprimento(traj),
        erroMax: erro.reduce((m, e) => Math.max(m, e), 0),
        bateu: colidiu(traj, obstaculos), tSim: (traj.length - 1) * DT,
      };
    }
    return { params: p, ini, obj, obstaculos, grade, planos };
  }

  return { SECOES, PARAMETROS, sortear, validar, calcular, comprimento };
})();

if (typeof module !== "undefined") module.exports = Engine;
