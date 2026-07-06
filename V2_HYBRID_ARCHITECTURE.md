# V2: Multi-Agent Interaction Model (Lotka-Volterra N×N)

## Motivação

A investigação de diversidade mostrou que:

> O landscape de parâmetros do GTPA é um pico agudo — qualquer desvio do ótimo piora o resultado. Não existe "ensemble heterogêneo" viável com o mesmo modelo.

O próximo salto exige uma **mudança de arquitetura**, não tuning de parâmetros.

A proposta: substituir a predição de aceleração da GTPA por um **sistema de forças multi-agente N×N**, onde cada jogador interage com todos os outros 21 jogadores + bola via forças de atração/repulsão/cooperação.

## Arquitetura

```
Estado atual (pos, vel para 22 jogadores + bola)
         │
         ▼
Para cada frame (1..30):
         │
    ┌────┴────┐
    │         │
  Bola:    Para cada agente i:
  Fluid      │
  Ball     Para cada agente j ≠ i:
  + Heun     │
             ├── relacao = classificar(i, j)
             │   │
             │   ├── PERSEGUIR: força += atração(i, j, peso)
             │   ├── DESVIAR:   força += repulsão(i, j, peso)
             │   ├── APOIAR:    força += cooperação(i, j, peso)
             │   └── IGNORAR:   força = 0
             │
             └── força += influência_da_bola(i)
                  │
                  ▼
             v_i += força × dt
             x_i += Heun(v_i)
         │
         ▼
Próximo frame
```

## Matriz de Interação A_ij

Cada par (i, j) tem um tipo de interação dinâmico:

| A_ij | Tipo | Comportamento | Quando |
|:----:|------|--------------|--------|
| +1 | Perseguir | Atração | Adversário com bola |
| -1 | Desviar | Repulsão | Marcador muito próximo (< 2m) |
| +0.5 | Apoiar | Cooperação | Companheiro livre |
| 0 | Ignorar | — | Adversário distante (> 5m) |

**A cada frame**, A_ij é recalculado baseado nas posições relativas, velocidades e posse de bola.

## Forças

### Perseguir (atração)

```python
def forca_perseguir(i, j, k=0.3):
    direcao = (pos_j - pos_i) / ||pos_j - pos_i||
    return k * direcao
```

### Desviar (repulsão)

```python
def forca_desviar(i, j, k=0.5, d_min=2.0):
    dist = ||pos_i - pos_j||
    if dist < d_min:
        direcao = (pos_i - pos_j) / dist
        return k * (d_min - dist) / d_min * direcao
    return 0
```

### Apoiar (cooperação)

```python
def forca_apoiar(i, j, k=0.15):
    # puxa em direção a uma posição intermediária
    vetor = (pos_j - pos_i) / ||pos_j - pos_i||
    return k * vetor
```

### Influência da bola

```python
def forca_bola(i, k=0.4):
    if sou_goleiro:
        return 0  # goleiro tem dinâmica própria
    direcao = (pos_ball - pos_i) / ||pos_ball - pos_i||
    return k * direcao
```

## Classificador de Relação

```python
def classificar(i, j, ball, possession):
    if j == ball:
        return INFLUENCIA_BOLA
    if possession[j] and j é adversário:
        return PERSEGUIR  # adversário com a bola
    if ||pos_i - pos_j|| < 2.0 and j é adversário:
        return DESVIAR  # muito perto
    if j é companheiro and ||pos_j - ball|| < ||pos_i - ball||:
        return APOIAR  # companheiro mais perto da bola
    if j é adversário and ||pos_i - pos_j|| > 5.0:
        return IGNORAR  # longe
    return IGNORAR
```

## Integração com GTPA (híbrido)

O GTPA não é descartado — ele vira **uma das fontes de força**:

```python
forca_total = (1 - w) * forca_lv + w * forca_gtpa
```

Onde `w` decai com o horizonte:
- Frames 1-5: w=1.0 (confia no GTPA, que é preciso no curto prazo)
- Frames 5-15: w decai linearmente para 0.3 (transição)
- Frames 15-30: w=0.3 (Lotka-Volterra domina, GTPA como correção)

Isso aproveita o melhor dos dois mundos: precisão neural no curto prazo + estabilidade física no longo prazo.

## Vantagens

1. **Sem drift neural**: as forças são grounded em física, não acumulam erro de rede
2. **Determinístico**: sem PF, sem partículas, sem ruído
3. **Interpretável**: cada força tem significado tático
4. **N-body nativo**: jogadores interagem entre si, não via GNN
5. **Rápido**: O(N²) com N=23 ~ 529 pares por frame, trivial

## Como calibrar as forças

Os parâmetros (k de cada força, distâncias limite) podem ser:
- Estimados dos dados de treino (distribuição de distâncias jogador-bola, jogador-jogador)
- Otimizados por busca em grade nos 3 cenários de test_old
- Aprendidos por regressão (se tivermos ground truth de aceleração)

## Próximos Passos

1. Implementar o simulador Lotka-Volterra N×N puro (sem GTPA)
2. Testar se consegue manter jogadores em campo por 30 frames
3. Comparar trajetórias com ground truth do test_old
4. Se viável, integrar com GTPA como fonte de força mista

## Motivação

O experimento de seleção de trajetórias (--trajectory_select) revelou algo fundamental:

> **Seleção entre hipóteses similares não adianta. O ganho vem de gerar hipóteses QUALITATIVAMENTE diferentes e só então selecionar.**

O weighted average do PF funciona como regularizador de ensemble porque as 32 partículas são correlacionadas (mesmo modelo, mesmo γ, mesmo intercept). Nesse regime, a média é ótima.

A V2 muda o paradigma:

```
V1: 1 modelo × 32 partículas → média ponderada
V2: K modelos × P partículas → seleção bipartido
```

## Arquitetura

```
Input (histórico)
         │
    ┌────┴────┐
    │         │
  GTPA     FeatureExtractor (vel, acc, ball-dist, formação)
    │         │
    │    Detector de Regime (play / pass / shot / drift)
    │         │
    └────┬────┘
         │
   Gerador de Hipóteses (K=4)
         │
    ┌────┼────┬────┐
    │    │    │    │
   H₁   H₂   H₃   H₄
  γ=0.6 γ=0.4 γ=0.8 γ=0.6
  Heun  Heun  Euler Heun
  int=y int=y int=n int=y+PN
         │
    ┌────┴────┐
    │         │
  Custo   Custo
  Físico  Regime
    │         │
    └────┬────┘
         │
  Bipartite Matching (trajetória completa)
         │
   Trajetória Final (30 frames × 12 agentes)
```

### Componentes

#### 1. Gerador de Hipóteses (K configurações)

Cada hipótese é UMA CONFIGURAÇÃO COMPLETA rodando o PF com weighted average internamente:

| Hipótese | γ | Integrator | Intercept | test_old | 2026 |
|----------|:-:|:----------:|:---------:|:--------:|:----:|
| H₁ (ref) | 0.8 | Heun | β=0.5 | 6.81m | **14.60m** |
| H₂ (leve) | 0.6 | Heun | β=0.5 | 6.85m | 14.74m |
| H₃ (inercial) | 0.4 | Heun | β=0.5 | 7.04m | 15.46m |
| H₄ (PN) | 0.8 | Heun | PN | — | — |

Idealmente K=4 ou K=8, cada uma usando 32/P = 8 partículas para manter 32 forward passes totais.

#### 2. Custo por Trajetória (30 frames)

```python
def trajectory_cost(traj, regime=None):
    """
    traj: (T, 23, 4)
    regime: regime detectado (opcional, para bônus)
    """
    # Física da bola: posição consistente com velocidade
    ball_drift = ||ball_pos_t - (ball_pos_{t-1} + ball_vel_{t-1})||

    # Suavidade dos jogadores: sem saltos de velocidade
    player_accel = ||v_players_t - v_players_{t-1}||

    # Coerência bola-jogador: distâncias mudam suavemente
    bp_change = |dist_ball_player_t - dist_ball_player_{t-1}|

    # Limites de campo
    oob = sum(players fora do campo)

    # Bônus de regime (se disponível):
    #   Se regime=SHOT espera-se ball_vel alta → penaliza H₂ (inercial fraca)
    #   Se regime=DRIFT espera-se ball_vel baixa → penaliza H₁ (γ=0.6 forte demais)

    return (ball_drift * 5 + player_accel * 1 +
            bp_change * 2 + oob * 10 + regime_bonus)
```

#### 3. Bipartite Selection

Entrada: K trajetórias completas (30 × 23 × 4)
Saída: 1 trajetória (menor custo acumulado)

```python
costs = [trajectory_cost(h, regime) for h in hypotheses]
best = argmin(costs)
```

O matching é trivial (seleção 1-de-K) porque cada hipótese é uma trajetória COMPLETA. A complexidade está em gerar hipóteses diversas, não em selecionar.

## Por que isso é diferente do que foi testado

O experimento que fiz (--trajectory_select) selecionava entre 32 partículas do MESMO modelo. O resultado foi 6.99m (> 6.85m) porque não havia diversidade real.

Na V2, cada hipótese é gerada por uma CONFIGURAÇÃO DIFERENTE:
- γ diferentes → comportamentos de bola radicalmente diferentes
- Integradores diferentes → propagação de erro diferente
- Intercept diferente → jogadores reagem à bola de forma diferente

Isso cria hipóteses qualitativamente distintas, onde a seleção faz sentido.

## Aperfeiçoamento: Bola como Sistema Híbrido (Logístico + Física)

### O Problema

O LV N×N funciona para jogadores porque eles **perseguem, desviam, apoiam** — forças contínuas bem modeladas por Lotka-Volterra.

A bola **não segue esse padrão**. Ela é um objeto passivo que sofre **transições discretas** (passes, chutes, desvios). Tentar modelar a bola com LV puro falha porque:

1. A bola não "persegue" ninguém — não há força natural de atração/repulsão
2. O fluid ball (γ) já modela bem a dinâmica livre
3. Faltam os eventos de kick que mudam a trajetória abruptamente

### A Solução: Mapa Logístico para Regime da Bola

O mapa logístico `x_{n+1} = r·x_n·(1-x_n)` controla a **intensidade do estado da bola**, não a posição:

| Regime | Faixa de x | r típico | Dinâmica |
|--------|:----------:|:--------:|----------|
| CONTROLLED | 0.0-0.3 | 1.0-2.5 | Bola segue o jogador mais próximo (possuidor) |
| FREE | 0.3-0.7 | 2.5-3.3 | Fluid Ball + Heun (γ=0.8) |
| PASS/SHOT | 0.7-1.0 | 3.3-4.0 | Impulso instantâneo na velocidade da bola |

O parâmetro `r` depende do contexto do jogo:
- `r` pequeno quando a bola está perto de um jogador (< 1m) e a velocidade é baixa
- `r` médio quando a bola está livre (distante de todos)
- `r` alto quando há indício de evento (mudança brusca de velocidade, jogador se aproximando rápido)

```python
def ball_regime(ball, players, x_t):
    # r_context baseado no estado do jogo
    nearest_dist = min(np.linalg.norm(ball.pos - p.pos) for p in players)
    ball_speed = np.linalg.norm(ball.vel)

    if nearest_dist < 1.0 and ball_speed < 0.5:
        r = 1.5  # controlled
    elif nearest_dist > 3.0:
        r = 3.0  # free
    else:
        r = 2.0 + ball_speed  # transition

    x_next = r * x_t * (1 - x_t)  # logistic map

    if x_next < 0.3:
        return CONTROLLED, x_next
    elif x_next < 0.7:
        return FREE, x_next
    else:
        return KICK, x_next
```

### Vantagem do Mapa Logístico

1. **Endógeno**: a transição de regime emerge do próprio estado da bola, não de um classificador externo
2. **Caos natural**: em `r > 3.57`, o mapa é caótico — modela a imprevisibilidade de passes e desvios
3. **Simplicidade**: uma única equação 1D substitui um detector de regime treinado
4. **Interpretável**: `x` é a "intensidade de evento" — baixo = controlado, alto = prestes a mudar

### Arquitetura Final V2

```
A cada frame:
  │
  ├── Jogadores: LV N×N (forças de atração/repulsão/cooperação)
  │     │
  │     ├── v_i += Σ_j A_ij · e^(-d_ij/σ) · direção_ij · dt
  │     ├── v_i *= damping
  │     ├── pos_i += Heun(v_i)
  │     └── field_clamp(pos_i)
  │
  ├── Bola: Sistema Híbrido
  │     │
  │     ├── x_{t+1} = r(ball, players) · x_t · (1 - x_t)
  │     │
  │     ├── if CONTROLLED:
  │     │     ball.vel = (nearest_player.pos - ball.pos) * 0.3
  │     │
  │     ├── elif FREE:
  │     │     ball.vel *= (1 - γ·dt) + noise
  │     │     ball.pos += Heun(vel)
  │     │
  │     └── elif KICK:
  │           ball.vel = direção_do_passe * kick_strength
  │           ball.pos += Heun(vel)
  │
  └── Avançar frame
```

### Relação com Regime Detection

O detector de regime (original da V1->V2) agora tem DOIS papéis:

1. **Gerar hipóteses**: usar o regime detectado para INSTANCIAR configurações (ex: se SHOT, incluir H₂ com γ baixo)
2. **Premiar/castear**: o regime serve como bônus no custo, favorecendo a hipótese mais coerente com o evento detectado

O detector pode ser simples (threshold em velocidade/aceleração) ou treinado (.rcl events → XGBoost).

## Próximos Passos

1. **Infra**: modificar `_stepwise_simulate` para rodar K configurações e armazenar K trajetórias
2. **Diversidade**: definir K=4 perfis (γ, integrator, intercept) que maximizem diferença semântica
3. **Custo**: implementar `_trajectory_cost` (já esboçado no PF) com pesos calibrados
4. **Regime**: treinar detector simples com features de velocidade/aceleração da bola
5. **Teste**: comparar seleção bipartido vs weighted average no test_old e 2026

## Risco

O ganho máximo teórico é limitado porque cada hipótese individualmente é pior que a média do ensemble. A seleção só ganha se a hipótese correta para o regime atual for significativamente melhor que a média das outras.

Dado que V1 já tem 14.74m e o teto deve estar em ~14.5m (limite da rede GTPA), o ganho esperado da V2 é de 0.2-0.5m.
