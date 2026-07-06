# V2: Heterogeneous Ensemble + Bipartite Trajectory Selection

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

| Hipótese | γ | Integrator | Intercept | Descrição |
|----------|:-:|:----------:|:---------:|----------|
| H₁ (ref) | 0.6 | Heun | β=0.5 | Baseline V1 (14.74m) |
| H₂ (inercial) | 0.4 | Heun | β=0.5 | Bola voa mais longe |
| H₃ (amortecido) | 0.8 | Euler | β=0.3 | Bola para rápido |
| H₄ (PN) | 0.6 | Heun | PN | Intercept proporcional |

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

## Relação com Regime Detection

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
