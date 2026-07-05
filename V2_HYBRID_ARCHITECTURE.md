# V2: Hybrid Regime-Aware Trajectory Prediction

## Arquitetura

```
Últimos 50 frames
        │
        ▼
Estimativa de velocidade e aceleração
        │
        ▼
Detector de Regime
        │
 ┌──────┼──────────┐
 │      │          │
 ▼      ▼          ▼
Linear FluidBall Lorenz
 │      │          │
 └──────┴──────────┘
        │
        ▼
Intercept Correction
        │
        ▼
Predição Final
```

O sistema é **recalculado a cada frame** — não gera uma trajetória única para 30 frames, mas sim um **Lorenz Local Adaptativo** passo a passo:

```
Frame t
     │
Detecta regime
     │
Calcula Lorenz local (se necessário)
     │
Atualiza bola
     │
Corrige jogadores (intercept)
     │
Frame t+1
```

Isso mantém a simplicidade (sem RNN), preserva a física do Fluid Ball e usa o componente não linear apenas quando ele acrescenta informação.

## Implementação

### BallState

```python
import numpy as np

class BallState:
    def __init__(self, pos, vel, acc):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.acc = np.asarray(acc, dtype=float)
```

### Detector de Regime

```python
from enum import Enum

class Regime(Enum):
    LINEAR = 0
    FLUID = 1
    CHAOTIC = 2

def detect_regime(ball):
    speed = np.linalg.norm(ball.vel)
    accel = np.linalg.norm(ball.acc)

    if speed < 0.3:
        return Regime.LINEAR
    if accel < 0.5:
        return Regime.FLUID
    return Regime.CHAOTIC
```

### Modelo Linear

```python
def linear_step(ball):
    ball.pos += ball.vel
    return ball
```

### Fluid Ball (γ)

```python
def fluid_step(ball, gamma=0.6):
    ball.vel *= gamma
    ball.pos += ball.vel
    return ball
```

### Lorenz (apenas perturbação)

O Lorenz **não substitui a trajetória** — apenas modifica a velocidade prevista:

```python
def lorenz_step(ball, sigma=10, rho=28, beta=8/3, dt=0.01):
    x, y = ball.vel
    z = 1.0
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    ball.vel += np.array([dx, dy]) * dt
    ball.pos += ball.vel
    return ball
```

### Atualização Automática

```python
def update_ball(ball):
    regime = detect_regime(ball)
    if regime == Regime.LINEAR:
        return linear_step(ball)
    if regime == Regime.FLUID:
        return fluid_step(ball)
    return lorenz_step(ball)
```

### Predição dos 30 Frames

```python
trajectory = []
for i in range(30):
    ball = update_ball(ball)
    trajectory.append(ball.pos.copy())
```

### Intercept dos Jogadores

```python
for player in players:
    direction = ball.pos - player.pos
    d = np.linalg.norm(direction)
    if d > 0:
        direction /= d
    player.pos += direction * player.speed
```

## Melhor Ainda: Pesos ao Invés de Troca

Ao invés de escolher um único modelo:

```python
linear = linear_predict(ball)
fluid = fluid_predict(ball)
lorenz = lorenz_predict(ball)

w1 = 0.2  # peso linear
w2 = 0.7  # peso fluid
w3 = 0.1  # peso lorenz

prediction = w1 * linear + w2 * fluid + w3 * lorenz
```

Os pesos podem depender de:
- Velocidade da bola
- Aceleração
- Distância ao jogador mais próximo
- Densidade de jogadores (Delaunay/Voronoi)

## Core Insight

O problema atual é que o **Fluid Ball/Lorenz** está sendo usado como um **modelo global**, enquanto o jogo de futebol é um sistema **híbrido**:

- **Regiões lineares/contínuas:** jogadores correndo, bola em posse, deslocamentos suaves
- **Regiões dinâmicas/discretas:** passe, chute, mudança brusca de direção, disputa pela bola

### Por que isso faz sentido

Os experimentos mostraram que:

- γ=0.6 funciona porque estabiliza o sistema
- O problema não é falta de caos; é que o caos aparece apenas em alguns momentos
- Aplicar um modelo caótico continuamente pode "inventar" dinâmica onde ela não existe
- Amortecer continuamente pode "matar" dinâmica onde ela deveria existir

### Vantagem científica

Deixa de tratar todo o jogo como caótico e passa a modelá-lo como um **sistema híbrido**, alternando entre dinâmica contínua e dinâmica não linear. Compatível com a linha de pesquisa baseada em modelos físicos e geométricos.

## Key Lessons from V1

1. **Intercept correction é responsável por ~60% do ganho** — manter na V2
2. **γ=0.6 constante > γ temporal** porque a predição da bola nunca fica confiável em 30 frames
3. **Bola = 5-19% da métrica** (1 de 12 agentes) — melhorar a bola sozinha dá ganho limitado
4. **Todas as 4 cenas oficiais são BALL-AT-FEET** — sem chutes no boundary
5. **O intercept precisa de um alvo estável** — γ alto mantém a bola numa região pequena

## Integrator Results

| Integrator | test_old | 2026 |
|------------|:--------:|:----:|
| legacy (network Euler) | 6.95m | 14.92m |
| Euler (pos_t + v*dt) | 7.60m | — |
| **Heun (RK2)** | **6.85m** | **14.74m** |
| Simpson 1/3 | 6.96m | 14.74m |
| AB2 | 7.67m | — |

Heun (RK2) = `x_{t+1} = x_t + (v_t + v_{t+1})/2 * dt`. Melhor integrador: reduz erro de integração sem custo computacional adicional.

## Dados para Treino do Detector

- `.rcg.gz`: full match logs (posições)
- `.rcl.gz`: event logs (kicks, fouls, goals com timestamps)
- Event CSVs em `ground-truth/`

## Next Steps

1. Parse `.rcl` para extrair kicks com frame numbers
2. Extrair features de velocidade/aceleração dos `.rcg`
3. Treinar classificador leve (XGBoost/MLP) para (features → kick)
4. Integrar no particle_filter.py como `use_event_aware`
5. Avaliar no test_old, depois one-shot no 2026
